#!/usr/bin/python3
import numpy as np
import rospy
from rosneuro_msgs.msg import NeuroFrame
from scipy.io import loadmat
from scipy.signal import butter, lfilter, lfilter_zi
import rosbag
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from Filter import FilterChain

RECORD_BAGS = False

class FilterChainNode():
    def __init__(self) -> None:
        self.node_name = 'filterchain_node'

    def setup(self):
        # Init the node        
        rospy.init_node(self.node_name)
        hz = rospy.get_param('rate', 16)
        self.rate = rospy.Rate(hz)
        # Init Subscriber and Publisher
        rospy.Subscriber('neurodata', NeuroFrame, lambda x: self.callback(x))
        self.pub = rospy.Publisher('neurodata_filtered', NeuroFrame, queue_size=1)
        # Setup the filter chain
        cfg = rospy.get_param(rospy.get_param(f'/{self.node_name}/configname'))
        self.filter_chain=FilterChain(cfg)

        self.new_data = False

    def run(self):
        if RECORD_BAGS:
            self.raw_bag = rosbag.Bag('/home/curtaz/Neurorobotics/rawdata.bag', 'w')
            self.filtered_bag = rosbag.Bag('/home/curtaz/Neurorobotics/filtdata.bag', 'w')
        # Spin until shutdown
        while not rospy.is_shutdown():
            # Wait until new data arrives
            if self.new_data:
                eeg = np.array(self.current_frame.eeg.data).reshape(self.current_frame.eeg.info.nsamples,self.current_frame.eeg.info.nchannels)
                eeg_data = self.filter_chain(eeg) 
                self.current_frame.eeg.data = eeg_data.flatten().tolist()
                self.pub.publish(self.current_frame)
                self.new_data = False
                if RECORD_BAGS:
                    new_msg = Float32MultiArray()
                    new_msg.layout.dim = [MultiArrayDimension()]
                    new_msg.data = self.current_frame.eeg.data
                    self.raw_bag.write('raw',new_msg)

                    new_msg = Float32MultiArray()
                    new_msg.layout.dim = [MultiArrayDimension()]
                    new_msg.data = eeg_data.flatten().tolist()
                    self.filtered_bag.write('filtered',new_msg)   
            self.rate.sleep()

        if RECORD_BAGS:
            self.raw_bag.close()
            self.filtered_bag.close()

    def callback(self,data: NeuroFrame):
        # Save the new data
        self.current_frame = data
        self.new_data = True

  
def main():
    filter_chain_node = FilterChainNode()
    filter_chain_node.setup()
    filter_chain_node.run()

if __name__ == '__main__':
  main()