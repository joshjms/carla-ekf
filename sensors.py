import carla
import math
import numpy as np
import xml.etree.ElementTree as ET
import weakref

class IMU:
    def __init__(self, world, vehicle):
        IMU_NOISE_ACCEL_STDDEV_X = 5e-5
        IMU_NOISE_ACCEL_STDDEV_Y = 5e-5
        IMU_NOISE_ACCEL_STDDEV_Z = 5e-5
        IMU_NOISE_GYRO_STDDEV_X = 5e-5
        IMU_NOISE_GYRO_STDDEV_Y = 5e-5
        IMU_NOISE_GYRO_STDDEV_Z = 5e-5

        IMU_FREQ = 2000

        imu_bp = world.get_blueprint_library().find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', str(1/IMU_FREQ))
        imu_bp.set_attribute('noise_accel_stddev_x', str(IMU_NOISE_ACCEL_STDDEV_X))
        imu_bp.set_attribute('noise_accel_stddev_y', str(IMU_NOISE_ACCEL_STDDEV_Y))
        imu_bp.set_attribute('noise_accel_stddev_z', str(IMU_NOISE_ACCEL_STDDEV_Z))
        imu_bp.set_attribute('noise_gyro_stddev_x', str(IMU_NOISE_GYRO_STDDEV_X))
        imu_bp.set_attribute('noise_gyro_stddev_y', str(IMU_NOISE_GYRO_STDDEV_Y))
        imu_bp.set_attribute('noise_gyro_stddev_z', str(IMU_NOISE_GYRO_STDDEV_Z))

        self.sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)

    def listen(self, queue):
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: enqueue_data(
            weak_self, 
            data, 
            queue
        ))

class GNSS:
    def __init__(self, world, vehicle):
        GNSS_NOISE_LAT_STDDEV = 5e-10
        GNSS_NOISE_LON_STDDEV = 5e-10
        GNSS_NOISE_ALT_STDDEV = 5e-10

        GNSS_FREQ = 10

        gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', str(1/GNSS_FREQ))
        gnss_bp.set_attribute('noise_alt_stddev', str(GNSS_NOISE_ALT_STDDEV))
        gnss_bp.set_attribute('noise_lat_stddev', str(GNSS_NOISE_LAT_STDDEV))
        gnss_bp.set_attribute('noise_lon_stddev', str(GNSS_NOISE_LON_STDDEV))

        self.sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)

    def listen(self, queue):
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda data: enqueue_data(
            weak_self, 
            data,
            queue
        ))
    
def enqueue_data(weak_self, data, queue):
    self = weak_self()
    if not self:
        return

    queue.put(data)