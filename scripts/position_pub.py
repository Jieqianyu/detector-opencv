#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray

import cv2
import time

from detector import Detector
from utils import CameraCalibration, AverageMeter, Cap


class CenterPub:
    def __init__(self,):
        rospy.init_node('position_pub', anonymous=True)

        self.pub_pos = rospy.Publisher('obj_world_position',  Float32MultiArray, queue_size=10)
        self.pub_vel = rospy.Publisher('obj_world_velocity',  Float32MultiArray, queue_size=10)
        self.detector = Detector()
        self.vs = Cap(source=rospy.get_param('~source'))
        self.cal = CameraCalibration()
        self.interval = 15
        self.position = None
        self.freq = 10

    def run(self,):
        rate = rospy.Rate(self.freq)
        delta_x = AverageMeter(max_len=self.interval)
        delta_y = AverageMeter(max_len=self.interval)
        while not rospy.is_shutdown():
            _, frame = self.vs.read()
            if frame is None or frame is []:
                continue
            # get position
            _, _, frame_position = self.detector.detect(frame)
            world_position = self.cal.frame_to_world(frame_position)
            if world_position is None:
                continue
            world_position = world_position.squeeze()
            if self.position is None:
                self.position = world_position

            # update position
            delta_x.update(world_position[0] - self.position[0])
            delta_y.update(world_position[1] - self.position[1])
            self.position = world_position
            
            # message
            position_msg = Float32MultiArray()
            position_msg.data.append(world_position[0])
            position_msg.data.append(world_position[1])

            velocity_msg = Float32MultiArray()
            velocity_msg.data.append(delta_x.avg*self.freq)
            velocity_msg.data.append(delta_y.avg*self.freq)

            # publish message
            self.pub_pos.publish(position_msg)
            self.pub_vel.publish(velocity_msg)
            rate.sleep()

if __name__ == "__main__":
    center_pub = CenterPub()
    center_pub.run()

