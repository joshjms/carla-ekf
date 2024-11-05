import os
import carla
import time
from queue import Queue

from sensors import IMU, GNSS

class Car:
    def __init__(self, world, client, spawn_point):
        self.world = world
        self.client = client

        vehicle_bp = world.get_blueprint_library().filter('vehicle.dodge.charger_2020')[0]
        self.vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(self.vehicle)

        self.imu = IMU(world, self.vehicle)
        self.gnss = GNSS(world, self.vehicle)

        self.imu_queue = Queue()
        self.gnss_queue = Queue()

        self.imu.listen(self.imu_queue)
        self.gnss.listen(self.gnss_queue)

    def start(self):
        '''
        Start the car autopilot
        '''
        self.vehicle.set_autopilot(True)

    def stop(self):
        '''
        Stop the car autopilot
        '''
        self.vehicle.set_autopilot(False)
        time.sleep(15)

    def get_id(self):
        '''
        Get car actor ID as defined by CARLA
        '''
        return self.vehicle.id

    def get_location(self, snapshot):
        '''
        Get car location in the world for a specific frame
        '''
        return snapshot.find(self.get_id()).get_transform().location
    
    def get_orientation(self, snapshot):
        '''
        Get car orientation in the world for a specific frame
        '''
        return snapshot.find(self.get_id()).get_transform().rotation
        
    def get_imu_data(self, frame):
        '''
        Get IMU data for a specific frame
        '''
        while not self.imu_queue.empty():
            imu_data = self.imu_queue.get()

            if imu_data.frame == frame:
                self.imu_queue.task_done()
                return imu_data

            self.imu_queue.task_done()

        return None
    
    def get_gnss_data(self, frame):
        '''
        Get GNSS data for a specific frame
        '''
        while not self.gnss_queue.empty():
            gnss_data = self.gnss_queue.get()

            if gnss_data.frame == frame:
                self.gnss_queue.task_done()
                return gnss_data

            self.gnss_queue.task_done()

    def destroy(self):
        self.imu.sensor.destroy()
        self.gnss.sensor.destroy()
        self.vehicle.destroy()