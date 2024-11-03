import carla
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

from camera import WorldCamera
from car import Car
from ekf import ExtendedKalmanFilter
from utils import gnss_to_xyz, get_latlon_ref, euler_to_quaternion

class Environment:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()

        self.world_camera = WorldCamera(self.world)

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
        self.show_est_traj_fig()
        self.show_est_traj_fig_2d()
        self.show_world_camera()

    def cleanup(self):
        try:
            self.car.destroy()
            self.world_camera.destroy()
        except:
            pass

    def show_est_traj_fig(self):
        est_traj_fig = plt.figure()
        ax = est_traj_fig.add_subplot(111, projection='3d')

        p_est = self.ekf.get_estimates()[0]

        ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
        ax.plot(self.true_p[:,0], self.true_p[:,1], self.true_p[:,2], label='Ground Truth')
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.set_zlabel('Up [m]')
        ax.set_title('Ground Truth and Estimated Trajectory')
        ax.set_xlim(-250, 250)
        ax.set_ylim(-250, 250)
        ax.set_zlim(-2, 2)
        ax.set_xticks([-250, -200, -150, -100, -50, 0, 50, 100, 150, 200])
        ax.set_yticks([-250, -200, -150, -100, -50, 0, 50, 100, 150, 200])
        ax.set_zticks([-2, -1, 0, 1, 2])
        ax.legend(loc=(0.62,0.77))
        ax.view_init(elev=45, azim=-50)
        plt.show()

    def show_est_traj_fig_2d(self):
        est_traj_fig = plt.figure()
        ax = est_traj_fig.add_subplot(111)

        p_est = self.ekf.get_estimates()[0]

        ax.plot(p_est[:,0], p_est[:,1], label='Estimated')
        ax.plot(self.true_p[:,0], self.true_p[:,1], label='Ground Truth')
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.set_title('Ground Truth and Estimated Trajectory')
        ax.set_xlim(-250, 250)
        ax.set_ylim(-250, 250)
        ax.set_xticks([-250, -200, -150, -100, -50, 0, 50, 100, 150, 200])
        ax.set_yticks([-250, -200, -150, -100, -50, 0, 50, 100, 150, 200])
        ax.legend(loc=(0.62,0.77))

        plt.show()

    def show_world_camera(self):
        image_folder = 'output'
        image_files = os.listdir(image_folder)
        print(image_files)
        
        images = []
        for filename in image_files:
            images.append(imageio.imread(os.path.join(image_folder, filename)))
        imageio.mimsave('world_cam.gif', images)

        


def create_env():
    return Environment()