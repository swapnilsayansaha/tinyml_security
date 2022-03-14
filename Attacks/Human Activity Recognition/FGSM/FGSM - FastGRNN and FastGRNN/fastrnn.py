#95%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import helpermethods
import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import sys
from tensorflow.compat.v1.keras.layers import Dense, Input, RNN
from tensorflow.compat.v1.keras.models import Model, Sequential, load_model
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1.keras.losses import MSE

from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  
config.log_device_placement = True  
sess2 = tf.compat.v1.Session(config=config)
set_session(sess2) 

tf.disable_eager_execution()
#FastRNN and FastGRNN imports
from rnn import FastGRNNCellKeras, FastRNNCellKeras
from data_utils import *


f = '/home/nesl/209as_sec/human_act/Data/Activity_Dataset/'
model_dir = 'trained_models/'
window_size = 550
stride = 50

X_tr, Y_tr, X_test, Y_test = import_auritus_activity_dataset(dataset_folder = f, 
                                use_timestamp=False, 
                                shuffle=True, 
                                window_size = window_size, stride = stride, 
                                return_test_set = True, test_set_size = 300)
print(X_tr.shape)
print(Y_tr.shape)
print(X_test.shape)
print(Y_test.shape)

Xtrain = X_tr
Ytrain = Y_tr
Xtest = X_test
Ytest = Y_test
numClasses = Y_tr.shape[1]
dataDimension = Xtrain[0].shape
channels = Xtrain.shape[2]

cell = "FastRNN" # Choose between FastGRNN, FastRNN  ###, UGRNN, GRU and LSTM

inputDims = [window_size, channels] #features taken in by RNN in one timestep
hiddenDims = 32 #hidden state of RNN

totalEpochs = 300
batchSize = 100

learningRate = 0.01
decayStep = 200
decayRate = 0.1

outFile = None #provide your file, if you need all the logging info in a file

#low-rank parameterisation for weight matrices. None => Full Rank
wRank = None 
uRank = None 

#Sparsity of the weight matrices. x => 100*x % are non-zeros
#Note: Sparsity inducing is not supported in this code. 
sW = 1.0 
sU = 1.0

#Non-linearities for the RNN architecture. Can choose from "tanh, sigmoid, relu, quantTanh, quantSigm"
update_non_linearity = "tanh"
gate_non_linearity = "sigmoid"


FastCell = FastRNNCellKeras(hiddenDims, update_non_linearity=update_non_linearity,
                           wRank=wRank, uRank=uRank)
                           
model = load_model('fastrnn.h5', custom_objects={'FastRNNCellKeras':FastCell})
model.compile(run_eagerly=False,experimental_run_tf_function=False,optimizer=Adam(), loss='categorical_crossentropy')


def fgsm_attack(model, image, label, eps):
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image)
        loss = MSE(label, pred)
        gradient = tape.gradient(loss, image)
        signedGrad = tf.sign(gradient)
        adversary = (image + (signedGrad * eps))
        return adversary
        
eps = [0.2]
#eps = [0.3,0.5,0.7,0.9,1.0,1.5,2.0,5.0,10.0,15.0,20.0,30.0,40.0,50.0]
accu_num = []
for item in eps:
    countadv = 0
    for i in range(len(Xtest)):
        act = Xtest[i,:,:].reshape(1,550,6)
        label = Ytest[i,:]
        actPred = model.predict(act)
        actPred = actPred.argmax()
        adversary = fgsm_attack(model,act, label, eps=item)
        pred = model.predict(adversary,steps=1,verbose=False)
        adversaryPred = pred[0].argmax()
        if actPred == adversaryPred:
            countadv += 1
    #print("Adversarial accuracy : ", countadv / len(X_test))
    accu_num.append(countadv / len(X_test)) ;
    
print(accu_num)
    
