from __future__ import division
import sys
# cv2_WRONG_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
# if cv2_WRONG_PATH in sys.path:
#     sys.path.remove(cv2_WRONG_PATH)


import cv2
import numpy as np
import time

from utils import xywh2xyxy, scale_bbox, show_img


class Detector(object):
    def __init__(self, shape=(640, 480)):
        self.window_name = 'Detection'
        self.debug = False
        self.box_type = 'min_area' # bbox or min_area
        self.contours_thres = (0, 150) # (25, 80) # select contours with number of pixels in (25, 80)
        self.view_result = True
        self.shape = shape

        # Parameters for HSV
        # (36, 202, 59, 71, 255, 255)    # Green
        # (18, 0, 196, 36, 255, 255)  # Yellow
        # (89, 0, 0, 125, 255, 255)  # Blue
        self.icol = (0, 156, 80, 10, 255, 255)
        self.hsv_names = ('low_hue', 'low_sat', 'low_val', 'high_hue', 'high_sat', 'high_val')

        if self.debug:
            cv2.namedWindow(self.window_name)
            self.create_trackbar()

    def plot_img(self, frame, box):
        dst = frame.copy()
        H, W, _ = dst.shape
        line_width = int(0.005 * (H+W) / 2)
        if box is not None:
            if self.box_type == 'bbox':
                x1, y1, x2, y2 = box
                dst = cv2.rectangle(dst, (x1, y1), (x2, y2), (255, 0, 0), line_width)
            elif self.box_type == 'min_area':
                dst = cv2.drawContours(dst, [box], 0, (0, 0,255), line_width)
            else:
                raise TypeError('unsupported box type %s' % self.box_type)

        return dst

    def create_trackbar(self,):
        for i in range(len(self.hsv_names)):
            cv2.createTrackbar(self.hsv_names[i], self.window_name, self.icol[i], 255, lambda x: x) 
    
    def get_trackbar_value(self,):
        hsv_thres_values = [] # ['low_hue', 'low_sat', 'low_val', 'high_hue', 'high_sat', 'high_val']
        for i in range(len(self.hsv_names)):
            hsv_thres_values.append(cv2.getTrackbarPos(self.hsv_names[i], self.window_name))

        return hsv_thres_values

    def detect(self, frame):
        '''
        input: frame
        output: dst, box(xyxy(bbox) or 4x2(min_area)), center(xy)
        '''
        if frame is None or frame is []:
            raise TypeError('No frame input')
        # Resize the frame
        shape = self.shape
        img0 = np.copy(frame)
        H, W, C = frame.shape
        scale_factor = np.array([shape[0]/W, shape[1]/H])

        frame = cv2.resize(frame, shape) # 300,400,3
        
        # Gaussian Blur
        frame_gaussian = cv2.GaussianBlur(frame, (7, 7), 0)
        
        # RGB to HSV
        frame_hsv = cv2.cvtColor(frame_gaussian, cv2.COLOR_BGR2HSV)
        
        # Get mask according to HSV
        hsv_thres_values = self.get_trackbar_value() if self.debug else list(self.icol)
        mask = cv2.inRange(frame_hsv, np.array(hsv_thres_values[:3]), np.array(hsv_thres_values[3:]))
        
        # Median filter
        mask_f = cv2.medianBlur(mask, 5)

        # Morphology for three times
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_m = cv2.morphologyEx(mask_f, cv2.MORPH_CLOSE, kernel)
        mask_m = cv2.morphologyEx(mask_m, cv2.MORPH_OPEN, kernel)
        
        # Get Contours of The Mask
        box = None # xyxy
        center = None # xy
        _, contours, _= cv2.findContours(mask_m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_list = [cnt for cnt in contours if self.contours_thres[0]<len(cnt)<self.contours_thres[1]]
        if cnt_list:
            cnt = max(contours, key=lambda x: x.shape[0]) # mutiple contours, choose the biggest one
        
            if self.box_type == 'bbox':
                # Get Bounding Box
                box = np.int0(scale_bbox(xywh2xyxy(cv2.boundingRect(cnt)), 1/scale_factor)) # xyxy
                print(box)
                center = np.int0(np.array([(box[0] + box[2])/2, (box[1] + box[3])/2]))
            elif self.box_type == 'min_area':
                # Get Minimum Area Box
                rect = cv2.minAreaRect(cnt) # center(x, y), (width, height), angle of rotation
                box = cv2.boxPoints(rect) # (4, 2)
                # scale box
                box = box / scale_factor
                center = np.sum(box, axis=0)/4
                box, center = np.int0(box), np.int0(center)
            else:
                raise TypeError('unsupported box type %s' % self.box_type)
        
        # Result
        dst = self.plot_img(img0, box)
        # view result
        if self.view_result:
            show_img(self.window_name, cv2.resize(dst, shape))
            # show_img(self.window_name, dst

        return dst, box, center


if __name__ == "__main__":
    is_save =  False
    src_path = "test1.jpg"
    process_shape = (640, 480)
    detector = Detector(process_shape)
    frame = cv2.imread(src_path, cv2.COLOR_BGR2RGB)
    # cap = cv2.VideoCapture('test2.mp4')
    while True:
        # _, frame = cap.read()
        if frame is None:
            break
        detector.detect(frame)
        time.sleep(0.01)
    # Save
    if is_save:
        dst_path = 'result_' + src_path.split('.')[0] + '.png'
        cv2.imwrite(dst_path, dst)
    
