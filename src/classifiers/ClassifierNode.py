#!/usr/bin/python3
import numpy as np
import rospy
import torch
from rosneuro_msgs.msg import (NeuroDataFloat, NeuroDataInfo, NeuroDataInt32,
                               NeuroFrame, NeuroOutput)

from Classifier import NeuralClassifier

class ClassifierNode():
    def __init__(self):
        self.node_name = 'predict_neural'

    def setup(self):
        self.new_data = False
        self.seq = 0
        # Init the node
        rospy.init_node(self.node_name)
        hz = rospy.get_param('rate', 16) # data.sr / nsample
        self.rate = rospy.Rate(hz)

        # Setup the Publisher
        self.pub = rospy.Publisher('neuroprediction', NeuroOutput, queue_size=1)
        
        # Setup the Subscriber
        rospy.Subscriber('neurodata_filtered', NeuroFrame, lambda x: self.callback(x))
        
        # Setup the classifier
        # cfg = rospy.get_param(rospy.get_param(f'/{self.node_name}/configname'))
        modelpath = rospy.get_param(f'/{self.node_name}/model_path')
        self.classes = [int(e) for e in rospy.get_param(f'/{self.node_name}/classes').strip('[]').split(',')]
        self.classifier = NeuralClassifier.from_config(modelpath)

    def run(self):
        # Central Loop
        while not rospy.is_shutdown():
            # Wait until new data arrives
            if self.new_data:
                pred,prob = self.buffered_prediction(self.current_frame)
                self.send_new_message(pred,prob)  
                self.new_data = False
            self.rate.sleep()

    def callback(self,data: NeuroFrame):
        # Save the new data
        self.current_frame = data
        self.new_data = True

    def send_new_message(self,pred,prob):
        # Starting from the old message generate the new one
        new_msg = NeuroOutput()
        info = NeuroDataInfo()
        # new_msg.decoder.classes = self.classifier.get_classes()
        new_msg.decoder.classes = self.classes
        new_msg.hardpredict = NeuroDataInt32(info,pred)
        new_msg.softpredict = NeuroDataFloat(info,prob)
        self.pub.publish(new_msg)

    def buffered_prediction(self,data: NeuroFrame):
        # INITIALIZATION
        if(self.seq==0):
            # Create a zero filled matrix
            self.buffer  = np.zeros((data.sr,data.eeg.info.nchannels))
            
        # UPDATE BUFFER
        # New data 
        eeg_data = np.array(data.eeg.data).reshape((data.eeg.info.nsamples, data.eeg.info.nchannels)) #all channels
        # Remove the old data from the buffer
        self.buffer = np.delete(self.buffer,[index for index in range(data.eeg.info.nsamples)], axis=0)
        # Add the new data to the buffer
        self.buffer = np.vstack((self.buffer, eeg_data))

        # Update the sequence number
        self.seq += 1
        
        to_pred = torch.from_numpy(self.buffer.astype(np.float32)).T.reshape(1,1,data.eeg.info.nchannels,data.sr)
        return self.classifier(to_pred)

 

def main():
    classifier = ClassifierNode()
    classifier.setup()
    classifier.run()

if __name__ == '__main__':
  main()
