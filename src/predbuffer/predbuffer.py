#!/usr/bin/python3

import rospy
import numpy as np
from rosneuro_msgs.msg import NeuroDataFloat, NeuroDataInfo, NeuroDataInt32,NeuroOutput
from std_msgs.msg import Int32

STOP = 0
RUNNING = 1

class BufferedPrediction:
    def __init__(self,n_classes,buffer_size):
        self.n_classes = n_classes
        self.buffer_size = buffer_size
        self.inc = 1/buffer_size
        self.reset_buffers()
        
    def __call__(self,data):
        pred = np.array(data.hardpredict.data)
        self.soft_buffer[pred == 1] += self.inc
        self.soft_buffer[pred != 1] -= self.inc
        self.soft_buffer[self.soft_buffer >= 1] = 1
        self.soft_buffer[self.soft_buffer <= 0] = 0

        self.hard_buffer[self.soft_buffer==1] = 1
        return self.hard_buffer,self.soft_buffer
    
    def reset_buffers(self):
        self.soft_buffer = np.zeros(self.n_classes,dtype=np.float32)
        self.hard_buffer = np.zeros(self.n_classes,dtype=int)

class BufferedPredictionNode:
    def __init__(self):
        self.node_name = 'buffered_prediction'

    def setup(self):
        self.new_data = False
        self.running_state = STOP
        self.seq = 0
        # Init the node
        rospy.init_node(self.node_name)
        hz = rospy.get_param('rate', 16) # data.sr / nsample
        self.rate = rospy.Rate(hz)

        # Setup the Publisher
        self.pub = rospy.Publisher('/integrator/neuroprediction', NeuroOutput, queue_size=1)
        
        # Setup the Subscriber
        rospy.Subscriber('/neuroprediction', NeuroOutput, lambda x: self.callback(x))
        rospy.Subscriber('/integrator/startStop',Int32 , lambda x: self.set_running_state(x))

        # Setup the classifier
        buffer_size = rospy.get_param(f'/{self.node_name}/buffer_size')
        n_classes = rospy.get_param(f'/{self.node_name}/n_classes')

        self.buffered_prediction = BufferedPrediction(n_classes,buffer_size)

    def set_running_state(self,state):
        new_state = state.data
        if self.running_state == STOP and new_state == RUNNING:
            self.running_state = RUNNING
            rospy.loginfo("Integration started")
        elif self.running_state == RUNNING and new_state == STOP:
            self.buffered_prediction.reset_buffers()
            self.running_state = STOP
            rospy.loginfo("Integration stopped. Buffers zeroed")
        else:
            rospy.logerr(f"An invalid command ({new_state}) has been sent.")


    def callback(self,data: NeuroOutput):
        # Save the new data
        self.current_frame = data
        self.new_data = True

    def send_new_message(self,old_msg,pred,prob):
        # Starting from the old message generate the new one
        new_msg = NeuroOutput()
        info = NeuroDataInfo()
        new_msg.decoder.classes = old_msg.decoder.classes
        new_msg.hardpredict = NeuroDataInt32(info,pred)
        new_msg.softpredict = NeuroDataFloat(info,prob)
        self.pub.publish(new_msg)

    def run(self):
        # Central Loop
        while not rospy.is_shutdown():
            # Wait until new data arrives
            if self.running_state == RUNNING and self.new_data:
                hard,soft = self.buffered_prediction(self.current_frame)
                self.send_new_message(self.current_frame,hard,soft)  
                self.new_data = False
                if (hard==1).any():
                    rospy.loginfo("Buffer Filles! Resetting...")
                    self.buffered_prediction.reset_buffers()
            self.rate.sleep()

def main():
    buffer = BufferedPredictionNode()
    buffer.setup()
    buffer.run()
if __name__ == '__main__':
    main()


