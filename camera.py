import carla

class WorldCamera:
    def __init__(self, world):
        cam_freq = 5

        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1280')
        cam_bp.set_attribute('image_size_y', '960')
        cam_bp.set_attribute('sensor_tick', str(1/cam_freq))

        cam_location = carla.Location(0, +20, 160)
        cam_rotation = carla.Rotation(roll=-90, pitch=-90, yaw=0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        self.sensor = world.spawn_actor(cam_bp, cam_transform)
        self.sensor.listen(lambda data: data.save_to_disk('output/%.6d.png' % data.frame))

    def destroy(self):
        self.sensor.destroy()

