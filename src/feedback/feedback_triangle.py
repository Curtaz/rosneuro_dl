#!/usr/bin/python3

import rospy

import cv2
import numpy as np

from rosneuro_msgs.msg import NeuroEvent,NeuroOutput
from std_srvs.srv import Empty

from events import *
import sys

COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0,0,255)
COLOR_GREEN = (0,255,0)
COLOR_BLUE = (255,0,0)
COLOR_WHITE = (255,255,255)
CLASS_COLORS = [COLOR_RED,COLOR_BLUE,COLOR_GREEN]

CLASSES = [771,770,769]

class SMRControl():
    def __init__(self):
        sys.argv = rospy.myargv(sys.argv)
        rospy.init_node('rosneuro_feedback')

        self.event_pub = rospy.Publisher("/events/bus", NeuroEvent, queue_size=1000)
        rospy.Subscriber("/integrator/neuroprediction", NeuroOutput, self.receive_probabilities)
        self.new_data = False
        self.classes = [int(c) for c in rospy.get_param('~classes',str(CLASSES)).strip('[]').split(',')]
        self.n_classes = len(self.classes)

        self.values = np.zeros(self.n_classes)

        self.window_width = rospy.get_param('~window_width',512)
        self.window_height = rospy.get_param('~window_height',512)
        self.triangle_side_length = rospy.get_param('~triangle_side',400)

        self.timings_begin = rospy.get_param('~timings_begin',1000)
        self.timings_fixation = rospy.get_param('~timings_fixation',1000)
        self.timings_cue = rospy.get_param('~timings_cue',1000)
        self.timings_feedback_update = rospy.get_param('~timings_feedback_update',1000)
        self.timings_boom = rospy.get_param('~timings_boom',1000)
        self.timings_end = rospy.get_param('~timings_end',1000)
        self.points = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 0],
                                [1, 1, 1],
                                [0.33, 0.33, 0.33]])
    def reset_bci(self):
        rospy.wait_for_service('/integrator/reset',timeout=0.5)
        resbci = rospy.ServiceProxy('/integrator/reset', Empty)
        try:
            resbci()
            return True
        except rospy.ServiceException as e:
            print("Service call failed: %s",e)
            return False
        
    def setup(self):
        self.gui = GUI(self.window_width,self.window_height,self.triangle_side_length)
        self.gui.setup()

    def receive_probabilities(self, msg):
        values = np.array(msg.softpredict.data)
        classes = msg.decoder.classes
        _,idx,_ = np.intersect1d(self.classes,classes,assume_unique=True,return_indices=True)
        print(idx,self.classes,classes)
        self.values = values[idx]
        self.new_data = True

    def run(self):
        exit = False
        self.new_data = False
        hit = False

        print("[smrbci] Protocol starts")
        img = self.gui.draw()
        cv2.imshow('Triangle', img)
        cv2.waitKey(self.timings_begin)
        publish_neuro_event(self.event_pub, Event.CFEEDBACK)
        self.reset_bci()

        while not exit:
            if self.new_data:
                self.gui.add_point(self.values)
                img = self.gui.draw()
                cv2.imshow('Triangle', img)
                self.new_data = False
                if any(self.values >= 1.0): hit = True
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit = True
            
            if hit:
                self.gui.add_point(self.values,color = CLASS_COLORS[self.values.argmax()],radius = 30)
                img = self.gui.draw()
                cv2.imshow('Triangle', img)

                publish_neuro_event(self.event_pub, Event.CFEEDBACK+Event.OFF)
                cv2.waitKey(self.timings_boom)
                self.reset_bci()
                publish_neuro_event(self.event_pub, Event.CFEEDBACK)
                hit = False

        print("[smrbci] Protocol ends")
        

class GUI():
    def __init__(self, width=512, height=512, triangle_side_length=400):
        self.width = width
        self.height = height
        self.triangle_side_length = triangle_side_length
        self.img = np.zeros((self.height, self.width, 3), np.uint8)

    def setup(self):
        self.triangle_height = int(np.sqrt(3) / 2 * self.triangle_side_length)
        self.center_x, self.center_y = self.width // 2, self.height // 2

        self.pts = np.array([
            [self.center_x, self.center_y - self.triangle_height // 2],
            [self.center_x + self.triangle_side_length // 2, self.center_y + self.triangle_height // 2],
            [self.center_x - self.triangle_side_length // 2, self.center_y + self.triangle_height // 2],
        ], np.int32)
        self.pts = self.pts.reshape((-1, 1, 2))

        self.centroid_x = int((self.pts[0][0][0] + self.pts[1][0][0] + self.pts[2][0][0]) / 3)
        self.centroid_y = int((self.pts[0][0][1] + self.pts[1][0][1] + self.pts[2][0][1]) / 3)
        
        cv2.polylines(self.img, [self.pts], isClosed=True, color=COLOR_WHITE, thickness=3)
        cv2.circle(self.img, tuple(self.pts [0,0]), radius=15, color=CLASS_COLORS[0], thickness=-1)
        cv2.circle(self.img, tuple(self.pts [1,0]), radius=15, color=CLASS_COLORS[1], thickness=-1)
        cv2.circle(self.img, tuple(self.pts [2,0]), radius=15, color=CLASS_COLORS[2], thickness=-1)

        self.add_point([0,0,0])

    def draw(self):
        img = self.img.copy()
        self.point.draw(img)
        return img
    
    def add_point(self, point_coords, color = COLOR_YELLOW,radius = 9):    
        self.point = Point(self.img,point_coords, self.triangle_side_length, self.centroid_x, self.centroid_y,color,radius)

class Point:
    def __init__(self, parent, coordinates, triangle_side_length, centroid_x, centroid_y,color,radius):
        self.coordinates = coordinates
        self.triangle_side_length = triangle_side_length
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.color = color
        self.radius = radius
    def project_point(self):
        p = self.coordinates
        return (int(self.triangle_side_length//2 * 2 / np.sqrt(3) * ( p[1] * np.sqrt(3)/2 - p[2] *np.sqrt(3)/2 )),
                int(self.triangle_side_length//2 * 2 / np.sqrt(3) * ( p[0] - p[1]*0.5 - p[2]*0.5 )))

    def draw(self, img):
        point = self.project_point()
        cv2.circle(img, (self.centroid_x + point[0], self.centroid_y - point[1]), radius=self.radius, color=self.color, thickness=-1)

if __name__ == "__main__":
    gui = SMRControl()
    gui.setup()
    gui.run()
    
    cv2.destroyAllWindows()
