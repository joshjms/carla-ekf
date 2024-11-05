import numpy as np

from rotations import Quaternion, skew_symmetric

class ExtendedKalmanFilter:
    '''
    Extended Kalman Filter Algorithm
    '''

    def __init__(self, p_init, v_init, q_init):
        self.prev_timestamp = 0

        #########################################################################################
        # Initialize the state and covariance matrix
        #########################################################################################
        self.p_est = np.empty((0, 3)) # position estimates
        self.v_est = np.empty((0, 3)) # velocity estimates
        self.q_est = np.empty((0, 4)) # orientation estimates as quaternions
        self.p_cov = np.zeros([1, 9, 9])  # covariance matrices at each timestep
        
        self.p_est = np.append(self.p_est, [p_init], axis=0)
        self.v_est = np.append(self.v_est, [v_init], axis=0)
        self.q_est = np.append(self.q_est, [q_init], axis=0)
        
        pos_var = 1
        orien_var = 1000
        vel_var = 1000
        self.p_cov[0, :3, :3] = np.eye(3) * pos_var
        self.p_cov[0, 3:6, 3:6] = np.eye(3) * vel_var
        self.p_cov[0, 6:, 6:] = np.eye(3) * orien_var
        
        #########################################################################################
        # Set the estimated sensor variances. We can tune these values to get better results.
        #########################################################################################
        self.var_imu_f = 0.10
        self.var_imu_w = 0.25
        self.var_gnss  = 0.01

        #########################################################################################
        # Calculate the noise Jacobians needed for the Kalman filter
        #########################################################################################
        
        # Motion model noise Jacobian
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)
        # Measurement model Jacobian
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)

    def measurement_update(self, y_k):
        sensor_var = self.var_gnss
        p_cov_check = self.p_cov[-2]
        p_check = self.p_est[-2]
        v_check = self.v_est[-2]
        q_check = self.q_est[-2]

        #########################################################################################
        # Calculate the Kalman gain
        #########################################################################################
        R = np.eye(3) * sensor_var
        K = p_cov_check @ self.h_jac.T @ np.linalg.inv(self.h_jac @ p_cov_check @ self.h_jac.T + R)

        #########################################################################################
        # Compute the error state
        #########################################################################################
        delta_x = K @ (y_k - p_check)

        #########################################################################################
        # Correct the predicted state
        #########################################################################################
        p_hat = p_check + delta_x[:3]
        v_hat = v_check + delta_x[3:6]
        q_hat = Quaternion(axis_angle=delta_x[6:]).quat_mult_left(q_check)

        #########################################################################################
        # Compute the corrected covariance
        #########################################################################################
        p_cov_hat = (np.eye(9) - K @ self.h_jac) @ p_cov_check

        self.p_cov[-1] = p_cov_hat
        self.p_est[-1] = p_hat
        self.v_est[-1] = v_hat
        self.q_est[-1] = q_hat
    
    def predict_state(self, imu_f, imu_w, timestamp):
        delta_t = 0.01
        if self.prev_timestamp != 0:
            delta_t = timestamp - self.prev_timestamp

        self.prev_timestamp = timestamp

        g = np.array([0, 0, 9.81])

        #########################################################################################
        # Update the state with a prediction step
        #########################################################################################
        


        C_ns = Quaternion(*self.q_est[-1]).to_mat()
        _p_est = self.p_est[-1] + delta_t * self.v_est[-1] + 0.5 * delta_t**2 * (C_ns @ imu_f + g)
        _v_est = self.v_est[-1] + delta_t * (C_ns @ imu_f + g)
        _q_est = Quaternion(axis_angle=imu_w * delta_t).quat_mult_left(self.q_est[-1])

        #########################################################################################
        # Linearize the motion model and compute the Jacobians
        #########################################################################################
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * delta_t
        F[3:6, 6:] = -skew_symmetric(C_ns @ imu_f) * delta_t
        Q = np.eye(6)
        Q[:3, :3] *= (delta_t**2) * self.var_imu_f
        Q[3:, 3:] *= (delta_t**2) * self.var_imu_w

        #########################################################################################
        # Propagate uncertainty
        #########################################################################################
        _p_cov = F @ self.p_cov[-1] @ F.T + self.l_jac @ Q @ self.l_jac.T

        #########################################################################################
        # Update the state and covariance
        #########################################################################################

        # Update the state
        self.p_est = np.append(self.p_est, [_p_est], axis=0)
        self.v_est = np.append(self.v_est, [_v_est], axis=0)
        self.q_est = np.append(self.q_est, [_q_est], axis=0)

        # Update the covariance
        self.p_cov = np.append(self.p_cov, [_p_cov], axis=0)

    def recv_imu_data(self, data):
        imu_f = np.array([data.accelerometer.x, data.accelerometer.y, data.accelerometer.z])
        imu_w = np.array([data.gyroscope.x, data.gyroscope.y, data.gyroscope.z])
        ts = data.timestamp

        try:
            self.predict_state(imu_f, imu_w, ts)
        except Exception as e:
            print('IMU update failed:', e)
            pass

    def recv_gnss_data(self, data):
        # Compute the Euclidean distance between the estimated position and the GNSS data
        # to decide whether to update the state or not
        # We can assume the GNSS data is compromised if the distance is greater than
        # THRESHOLD meters.
        distance = np.linalg.norm(self.p_est[-1] - data)
        print(distance)

        try:
            self.measurement_update(data)
        except Exception as e:
            print('GPS update failed:', e)
            pass

    def get_estimates(self):
        return self.p_est, self.v_est, self.q_est
    
    def get_position(self):
        return self.p_est[-1]
    
    def get_velocity(self):
        return self.v_est[-1]
    
    def get_orientation(self):
        return self.q_est[-1]
    
    def get_covariance(self):
        return self.p_cov[-1]