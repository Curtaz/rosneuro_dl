import os

import numpy as np
import rospy
import torch
from joblib import load
from neurorobotics_dl.models import EEGNet, PrototypicalModel
from rosneuro_msgs.msg import (NeuroDataFloat, NeuroDataInfo, NeuroDataInt32,
                               NeuroFrame, NeuroOutput)

# from std_msgs.msg import Float32MultiArray, MultiArrayDimension
# import rosbag


class Predictor():
    def __init__(self,):
        sbj_path = '/home/curtaz/Neurorobotics/models/all_subjects_20240208_162941'
        model_path = os.path.join(sbj_path,"model.pt")
        clf_path = os.path.join(sbj_path,"qda.joblib")
        mean_std_path = os.path.join(sbj_path,"mean_std.npz")

        self.net = PrototypicalModel(EEGNet( 32, 
                                        Chans = 32, 
                                        Samples = 512,
                                        dropoutRate = 0,
                                        kernLength = 256,
                                        F1 = 8, 
                                        D = 2, 
                                        F2 = 16),metric='cosine')

        self.net.load_state_dict(torch.load(model_path))
        self.clf = load(clf_path) 
        cc = np.load(mean_std_path)
        self.mu, self.sigma = cc['mu'], cc['sigma']

    def __call__(self,x):
        x = (x-self.mu)/self.sigma
        emb = self.net.compute_embeddings(x)
        pred = [0 for _ in range(len(self.clf.classes_))]
        pred[int(self.clf.predict(emb))] = 1
        probs = self.clf.predict_proba(emb)[0].tolist()
        return pred, probs

def callback(data: NeuroFrame):
    global new_data, current_frame
    # Save the new data
    current_frame = data
    new_data = True

def buffered_prediction(data: NeuroFrame):
    global buffer, seq, avgs, predictor
    
    # INITIALIZATION
    if(seq==0):
        # Create a zero filled matrix
        buffer  = np.zeros((data.sr,data.eeg.info.nchannels))
        
    # UPDATE BUFFER
    # New data 
    eeg_data = np.array(data.eeg.data).reshape((data.eeg.info.nsamples, data.eeg.info.nchannels)) #all channels
    # Remove the old data from the buffer
    buffer = np.delete(buffer,[index for index in range(data.eeg.info.nsamples)], axis=0)
    # Add the new data to the buffer
    buffer = np.vstack((buffer, eeg_data))

    # Update the sequence number
    seq = seq + 1

    # If the buffer is filled
    if(seq * data.eeg.info.nsamples >= data.sr):
        # Compute prediction:
        to_pred = torch.from_numpy(buffer.astype(np.float32)).T.reshape(1,data.eeg.info.nchannels,1,data.sr)
        return predictor(to_pred)

    return ([0,0],[0.,0.])

def generate_new_message(data, rate,old_message):
    # Starting from the old message generate the new one
    global pred_bag
    new_msg = NeuroOutput()
    pred,prob = data

    info = NeuroDataInfo()
    new_msg.hardpredict = NeuroDataInt32(info,pred)
    new_msg.softpredict = NeuroDataFloat(info,prob)

    # pred_msg = Float32MultiArray()
    # pred_msg.layout.dim = [MultiArrayDimension()]
    # pred_msg.data = pred
    # pred_bag.write('pred',pred_msg)

    # pred_msg = Float32MultiArray()
    # pred_msg.layout.dim = [MultiArrayDimension()]
    # pred_msg.data = prob
    # pred_bag.write('prob',pred_msg)
    return new_msg


def main():
    global new_data, current_frame, seq, predictor
    global pred_bag
    new_data = False
    seq = 0
    
    # Init the node
    rospy.init_node('predict_neural')
    hz = rospy.get_param('rate', 16) # data.sr / nsample
    rate = rospy.Rate(hz)

    # Setup the Publisher
    pub = rospy.Publisher('neuroprediction', NeuroOutput, queue_size=1)
    
    # Setup the Subscriber
    rospy.Subscriber('neurodata_filtered', NeuroFrame, callback)

    # Setup the classifier
    predictor = Predictor()
    predictor.net.to('cpu')

    # pred_bag = rosbag.Bag('/home/curtaz/Neurorobotics/predictions.bag', 'w')

    while not rospy.is_shutdown():
        # Wait until new data arrives
        if new_data:
            # s = rospy.Time.now()
            new_data = buffered_prediction(current_frame)

            new_msg = generate_new_message(new_data, hz,current_frame)
            pub.publish(new_msg)
            new_data = False
            # e = rospy.Time.now()
            # print((e-s).to_sec())


        rate.sleep()
    # pred_bag.close()
        
if __name__ == '__main__':
  main()
