import carla
import numpy as np
import matplotlib.pyplot as plt
import time

from car import Car
from ekf import ExtendedKalmanFilter
from utils import gnss_to_xyz, get_latlon_ref, euler_to_quaternion

class Environment:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()

        self.traffic_manager = self.client.get_trafficmanager(8000)

        self.lat_ref, self.lon_ref = get_latlon_ref(self.world)
        self.true_p = np.empty((0, 3))
        self.est_p = np.empty((0, 3))

        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = spawn_points[1]
        self.true_p = np.append(self.true_p, [[self.spawn_point.location.x, self.spawn_point.location.y, self.spawn_point.location.z]], axis=0)

        self.ekf = ExtendedKalmanFilter(
            p_init=np.array([self.spawn_point.location.x, self.spawn_point.location.y, self.spawn_point.location.z]),
            v_init=np.array([0, 0, 0]),
            q_init=np.array(
                euler_to_quaternion(
                    self.spawn_point.rotation.roll,
                    self.spawn_point.rotation.pitch,
                    self.spawn_point.rotation.yaw
                )
            ),
        )

    def run(self):
        self.p_true = np.array([])
        self.car = Car(self.world, self.client, self.spawn_point)

        self.traffic_manager.ignore_lights_percentage(self.car.vehicle, 100.0)
        self.traffic_manager.ignore_signs_percentage(self.car.vehicle, 100.0)

        self.car.start()

        while True:
            self.world.tick()
            snapshot = self.world.get_snapshot()
            frame = snapshot.frame
            timestamp = snapshot.timestamp
            
            imu_data = self.car.get_imu_data(frame)
            gnss_data = self.car.get_gnss_data(frame)

            if imu_data is not None:
                # We get the true position of the car to compare with our estimated position
                # to evaluate the performance of the EKF algorithm.
                tp = self.car.get_location(snapshot)
                self.true_p = np.append(self.true_p, [[tp.x, tp.y, tp.z]], axis=0)

                # We get the IMU data to update the EKF algorithm
                self.ekf.recv_imu_data(imu_data)
            
            if gnss_data is not None:
                # We get the GNSS data to update the EKF algorithm
                self.ekf.recv_gnss_data(gnss_to_xyz(self.lat_ref, self.lon_ref, gnss_data.latitude, gnss_data.longitude, gnss_data.altitude))

    def display(self):
        # Display the true position of the car
        true_x = self.true_p[:, 0]
        true_y = self.true_p[:, 1]
        true_z = self.true_p[:, 2]

        p_est, v_est, q_est = self.ekf.get_estimates()

        x_est = p_est[:, 0]
        y_est = p_est[:, 1]
        z_est = p_est[:, 2]

        plt.figure()
        plt.plot(true_x, true_y, label='True Path', color='blue', marker='o')
        plt.plot(x_est, y_est, label='Estimated Path', color='red', marker='x')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('True vs Estimated Path of the Car (2D)')
        plt.legend()

        plt.show()

        indices = range(len(true_x))
        plt.figure()
        plt.plot(indices, true_x, label='True X', color='blue', marker='o')
        plt.plot(indices, x_est, label='Estimated X', color='red', marker='x')

        plt.xlabel('Index')
        plt.ylabel('X Coordinate')
        plt.title('True vs Estimated X Coordinate for Each Index')
        plt.legend()

        plt.show()

    def cleanup(self):
        try:
            self.car.destroy()
        except:
            pass

def create_env():
    return Environment()