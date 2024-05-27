import rospy
from rosneuro_msgs.msg import NeuroEvent

class Event:
    OFF = 32768
    LEFT_HAND = 769
    RIGHT_HAND = 770
    BOTH_HANDS = 771
    BOTH_FEET = 773
    CFEEDBACK = 781
    FIXATION = 786
    TARGETHIT = 897
    TARGETMISS = 898
    EOG = 1024

def publish_neuro_event(pub,event):
	msg = NeuroEvent()
	msg.header.stamp = rospy.Time.now()
	msg.event = event
	pub.publish(msg)