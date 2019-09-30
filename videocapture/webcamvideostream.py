from threading import Thread

import cv2


# video stream for web-cameras or video files
class WebcamVideoStream:
    def __init__(self, src=0, vdelay_ms=0):
        self.stream = cv2.VideoCapture(src)
        self.vdelay = vdelay_ms
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()

        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()
            if self.vdelay > 0:
                cv2.waitKey(self.vdelay)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
