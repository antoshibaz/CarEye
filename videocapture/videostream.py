# video stream interface for all cameras
class VideoStream:
    def __init__(self, src=0, vdelay_ms=0, use_pi_camera=False, resolution=(320, 240),
                 framerate=30):
        if use_pi_camera:
            from .pivideostream import PiVideoStream
            self.stream = PiVideoStream(resolution=resolution,
                                        framerate=framerate)
        else:
            from .webcamvideostream import WebcamVideoStream
            self.stream = WebcamVideoStream(src=src, vdelay_ms=vdelay_ms)

    def start(self):
        return self.stream.start()

    def read(self):
        return self.stream.read()

    def stop(self):
        self.stream.stop()
