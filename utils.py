import numpy as np
import math
import xml.etree.ElementTree as ET

# From yan99033's code in Github
def gnss_to_xyz(gnss_lat_ref, gnss_long_ref, latitude, longitude, altitude):
    """Creates Location from GPS (latitude, longitude, altitude).
    This is the inverse of the _location_to_gps method found in
    https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
    
    Modified from:
    https://github.com/erdos-project/pylot/blob/master/pylot/utils.py
    """
    EARTH_RADIUS_EQUA = 6378137.0

    scale = math.cos(gnss_lat_ref * math.pi / 180.0)
    basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * gnss_long_ref
    basey = scale * EARTH_RADIUS_EQUA * math.log(
        math.tan((90.0 + gnss_lat_ref) * math.pi / 360.0))

    x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
    y = scale * EARTH_RADIUS_EQUA * math.log(
        math.tan((90.0 + latitude) * math.pi / 360.0)) - basey

    # This wasn't in the original method, but seems to be necessary.
    y *= -1

    return np.array([x, y, altitude])

def get_latlon_ref(world):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
    """
    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert pitch, yaw, roll (in radians) to a quaternion.
    Returns a unit quaternion as a numpy array of size 4: [qw, qx, qy, qz].
    """
    half_roll = roll / 2
    half_pitch = pitch / 2
    half_yaw = yaw / 2

    cos_r = np.cos(half_roll)
    sin_r = np.sin(half_roll)
    cos_p = np.cos(half_pitch)
    sin_p = np.sin(half_pitch)
    cos_y = np.cos(half_yaw)
    sin_y = np.sin(half_yaw)

    qw = cos_r * cos_p * cos_y + sin_r * sin_p * sin_y
    qx = sin_r * cos_p * cos_y - cos_r * sin_p * sin_y
    qy = cos_r * sin_p * cos_y + sin_r * cos_p * sin_y
    qz = cos_r * cos_p * sin_y - sin_r * sin_p * cos_y

    return np.array([qw, qx, qy, qz])
