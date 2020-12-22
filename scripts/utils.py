from __future__ import division
import time
from ctypes import *
import threading
import numpy as np
from collections import deque

# import sys
# cv2_WRONG_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
# if cv2_WRONG_PATH in sys.path:
#     sys.path.remove(cv2_WRONG_PATH)
import cv2

video_format = ['mp4', 'avi']


def scale_bbox(x, scale):
    # Scale [4] box from [x1, y1, x2, y2] or [x1, y1, w, h]
    y = np.copy(x)
    y[0] *= scale[0]
    y[2] *= scale[0]
    y[1] *= scale[1]
    y[3] *= scale[1]

    return y


def xyxy2xywh(x):
    # Convert [4] box from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = x[0]  # top-left
    y[1] = x[1]  # top-left
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height

    return y


def xywh2xyxy(x):
    # Convert [4] box from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[0] = x[0]  # top left x
    y[1] = x[1]  # top left y
    y[2] = x[0] + x[2]  # bottom right x
    y[3] = x[1] + x[3]  # bottom right y

    return y


def show_img_debug(name, frame):
    print("The current window is occupied by image '" + frame_name +
          "'. " + "Please press esc to shut down current window and move on.")
    start = time.time()
    while True:
        time_out = ((time.time() - start)) > 120
        cv2.imshow(name, frame)
        key = cv2.waitKey(33)
        if (key & 0xff) == 27 or time_out:
            if time_out:
                print("Time out! Shut down the window.")
            break


def show_img(name, frame):
    cv2.imshow(name, frame)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        raise StopIteration


class Cap():
    def __init__(self, source='0'):
        self.camera = False
        if str(source) == '0':
            self.camera = True
			# call API to query image
            self.dll = cdll.LoadLibrary("libJHCap.so")
            self.dll.CameraInit(0)
            self.dll.CameraSetResolution(0, 0, 0, 0)
            self.dll.CameraSetContrast.argtypes = [c_int, c_double]
            self.dll.CameraSetContrast(0, 1.15)
            self.buflen = c_int()
            self.width = c_int()
            self.height = c_int()
            self.dll.CameraGetImageSize(0, byref(self.width), byref(self.height))
            self.dll.CameraGetImageBufferSize(0, byref(self.buflen), 0x4)
            self.inbuf = create_string_buffer(self.buflen.value)
            self.img = None
            self.count = 0

            t = threading.Thread(target=Cap.update, args=(self,))
            t.start()
        elif source.split('.')[-1] in video_format:
			self.cv = cv2.VideoCapture(source)
			self.count = 0
        else:
            raise TypeError('No such video format.')
        

    def update(self,):
        while True:
            self.dll.CameraQueryImage(0, self.inbuf, byref(self.buflen), 0x104)
            arr = np.frombuffer(self.inbuf, np.uint8)
            self.img = np.reshape(
                arr, (self.height.value, self.width.value, 3))
            self.count += 1
            time.sleep(0.01)

    def read(self,):
		if self.camera:
			img0 = None if self.img is None else self.img.copy()
		else:
			ret, img0 = self.cv.read()
			self.count += 1
			if not ret:
				raise EOFError('Video End')
    
		return self.count, img0


class CameraCalibration(object):
    def __init__(self, verbo=False):
        self.s = 982.4449
        self.translation = np.array(
            [[-207.2518, -81.4856, 982.4449]]).reshape(3, 1)
        self.rotation = np.array([[0.9999, 0.0075, -0.0132],
                                  [-0.0075, 1.0, -0.0018],
                                  [0.0132, 0.0019, 0.9999]])
        self.intrinsic = np.array([[2.8826e3, 0, 963.2226],
                                   [0, 2.8854e3, 426.4272],
                                   [0, 0, 1]])

        self.verbo = verbo

    def calibration_to_frame(self, cal_position):
        frame_position = None
        if cal_position is not None:
            frame_position = np.dot(self.intrinsic, np.dot(
                self.rotation, cal_position) + self.translation)/self.s

        if self.verbo:
            print('calibration to frame(mm):', frame_position.squeeze())
        return frame_position

    def frame_to_world(self, pixel_xy):
        cal_position = None
        if pixel_xy is not None:
            frame_position = np.ones((3, 1))
            frame_position[0, 0], frame_position[1,
                                                 0] = pixel_xy[0], pixel_xy[1]

            intrinsic_inv, rotation_inv = np.linalg.inv(
                self.intrinsic), np.linalg.inv(self.rotation)
            cal_position = np.dot(rotation_inv, (np.dot(
                intrinsic_inv, self.s*frame_position) - self.translation))
        if self.verbo:
            print("frame to calibration(mm)", cal_position.squeeze())

        return self.calibration_to_world(cal_position)

    def calibration_to_world(self, cal_position):
        world_position = None
        if cal_position is not None:
            world_position = cal_position.copy()
            world_position[1] *= -1
            world_position /= 1000
            world_position[0] += 0.155
            world_position[1] -= 0.225
            if self.verbo:
                print("frame to  world(m):",
                      world_position.squeeze())

        return world_position


class AverageMeter(object):
    def __init__(self, max_len=50):
        self.max_len = max_len
        self.reset()

    def __len__(self):
        return len(self.deq)

    def reset(self):
        self.deq = deque(maxlen=self.max_len)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        for i in range(n):
            self.deq.append(val)
        self.val = val
        self.sum = np.sum(self.deq)
        self.count = len(self.deq)
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # cap = Cap()
    # cv2.namedWindow("s")
    # while 1:
    # 	_, img = cap.read()
    # 	print(count)
    # 	cv2.imshow("s", img)
    # 	key=cv2.waitKey(33) #change parameter according to frame rate, wait time = 1000/fps
    # 	if (key&0xff) == 27: #press ESC on image window to terminate the loop
    # 		break
    # cv2.destroyWindow("s")
    cal = CameraCalibration(verbo=True)
    cal.calibration_to_frame(np.zeros((3, 1)))
    cal.frame_to_world((355, 187))
