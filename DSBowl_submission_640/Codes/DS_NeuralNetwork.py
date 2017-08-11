import dicom
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import pandas
import collections
import scipy.misc
import skimage.feature
from PIL import Image
from random import shuffle
from skimage import io
from tqdm import tqdm,tnrange
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:

    
    def __init__(self, input_dim, hidden_dim2, hidden_dim, output_dim):

        
        self.theta1 = np.random.randn(input_dim, hidden_dim2) / np.sqrt(input_dim)
        self.bias1 = np.zeros((1, hidden_dim2))
        self.theta2 = np.random.randn(hidden_dim2, hidden_dim) / np.sqrt(hidden_dim*hidden_dim)
        self.bias2 = np.zeros((1, hidden_dim))
        self.theta3 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim*hidden_dim)
        self.bias3 = np.zeros((1, output_dim))
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):

        num_examples = np.shape(X)[0]
        z1 = np.dot(X,self.theta1) + self.bias1
        a1 = np.tanh(z1)
        z2 = np.dot(a1,self.theta2) + self.bias2
        a2 = np.tanh(z2)
        z3 = np.dot(a2,self.theta3) + self.bias3
        exp_z = np.exp(z3)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        one_hot_y = np.zeros((num_examples,np.max(y)+1))
        logloss = np.zeros((num_examples,))        
        for i in range(np.shape(X)[0]):
            one_hot_y[i,y[i]] = 1
            logloss[i] = -np.sum(np.log(softmax_scores[i,:]) * one_hot_y[i,:])
        data_loss = np.sum(logloss)
        return 1./num_examples * data_loss

    
    #--------------------------------------------------------------------------
 
    def predict(self,X):

        z1 = np.dot(X,self.theta1) + self.bias1
        a1 = np.tanh(z1)
        z2 = np.dot(a1,self.theta2) + self.bias2
        a2 = np.tanh(z2)
        z3 = np.dot(a2,self.theta3) + self.bias3
        exp_z = np.exp(z3)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        predictions = np.argmax(softmax_scores, axis = 1)
        return predictions
        
    #--------------------------------------------------------------------------
    def prob(self,X):

        z1 = np.dot(X,self.theta1) + self.bias1
        a1 = np.tanh(z1)
        z2 = np.dot(a1,self.theta2) + self.bias2
        a2 = np.tanh(z2)
        z3 = np.dot(a2,self.theta3) + self.bias3
        exp_z = np.exp(z3)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return softmax_scores
        
    #--------------------------------------------------------------------------
    
    
    def fit(self,X,y,num_epochs=10000,lr=0.01):
 

        for epoch in tqdm(range(0, num_epochs)):
            

            # Forward propagation
            z1 = np.dot(X,self.theta1) + self.bias1
            a1 = np.tanh(z1)
            z2 = np.dot(a1,self.theta2) + self.bias2
            a2 = np.tanh(z2)
            z3 = np.dot(a2,self.theta3) + self.bias3
            exp_z = np.exp(z3)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            # Backpropagation
            beta2 = np.zeros_like(softmax_scores)
            one_hot_y = np.zeros_like(softmax_scores)
            for i in range(X.shape[0]):
                one_hot_y[i,y[i]] = 1
            beta2 = softmax_scores - one_hot_y
            beta1 = np.dot(beta2, self.theta3.T) * (1 - np.power(a2, 2))
            beta0 = np.dot(beta1, self.theta2.T) * (1 - np.power(a2, 2))
                        
            # Compute gradients of model parameters
            dtheta3 = np.dot(a2.T, beta2)
            dbias3 = np.sum(beta2, axis=0)
            dtheta2 = np.dot(a1.T, beta1)
            dbias2 = np.sum(beta1, axis=0)
            dtheta1 = np.dot(X.T, beta0)
            dbias1 = np.sum(beta0, axis=0)
        
            # Gradient descent parameter update
            self.theta3 -= lr * dtheta3
            self.bias3 -= lr * dbias3
            self.theta2 -= lr * dtheta2
            self.bias2 -= lr * dbias2
            self.theta1 -= lr * dtheta1
            self.bias1 -= lr * dbias1
            
        return 0

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


datadir = 'D:\Artificial Intelligence\\noncancerous'
datadir1 = 'D:\Artificial Intelligence\cancerous'

imageset = [[[0]*841,[0]] for _ in range(len(os.listdir(datadir))+len(os.listdir(datadir1)))]
ij = 0
for c in os.listdir(datadir1):
    img = np.array(scipy.misc.imread(datadir1+'/'+ c,mode = 'L')).flatten()
    imageset[ij][0] =  [(i+ii) if i and ii else i or ii for (i,ii) in map(lambda x,y: (x,y),imageset[ij][0],img)]
    imageset[ij][1] = 1
    ij+=1
    
for nc in os.listdir(datadir):
    img = np.array(scipy.misc.imread(datadir+'/'+ nc,mode = 'L')).flatten()
    imageset[ij][0] =  [(i+ii) if i and ii else i or ii for (i,ii) in map(lambda x,y: (x,y),imageset[ij][0],img)]
    imageset[ij][1] = 0
    ij+=1

shuffle(imageset)

outset = [[0] for _ in range(len(os.listdir(datadir))+len(os.listdir(datadir1)))]
inset = [[0]*841 for _ in range(len(os.listdir(datadir))+len(os.listdir(datadir1)))]
for image in range(len(imageset)):
    inset[image] = [(i+ii) if i and ii else i or ii for (i,ii) in map(lambda x,y: (x,y),inset[image],imageset[image][0])]
    outset[image] = imageset[image][1]
inset = np.array(inset)
outset = np.array(outset)


input_dim = np.shape(inset)[1]
output_dim = np.max(outset) + 1
first_hidden_dim = 1000 #set number of hidden nodes
hidden_dim = 1000 #set number of hidden nodes
nnet = NeuralNetwork(input_dim,first_hidden_dim, hidden_dim, output_dim)
#Fit model
nnet.fit(inset,outset,lr = 0.000001)
y_pred = nnet.predict(inset)
y_prob = nnet.prob(inset)
#Compute accuracy and confusion matrix
acc = 0
con_mat = np.zeros((output_dim, output_dim))
for i in range(len(y_pred)):
    con_mat[y_pred[i], outset[i]] += 1
    if outset[i] == y_pred[i]:
        acc += 1
acc = acc/len(y_pred)

print ('ACCURACY: ', acc)
print 'CONFUSION MATRIX: \n', con_mat


datadir3 = 'D:\Artificial Intelligence\stage1_sample_submission.csv'
datadir4 = 'D:\Artificial Intelligence\\finalNodes'

posnames = ['X','Y']
data = pandas.read_csv(datadir3,names = posnames)
patname = list(data['X'])
patclass = list(data['Y'])
predds = []
for pat in patname:
    predd = []
    if(pat) in os.listdir(datadir4):
        for nod in os.listdir(datadir4+'/'+ pat):
            img = np.array(scipy.misc.imread(datadir4+'/'+ pat+'/'+nod ,mode = 'L')).flatten()
            nod_prob = nnet.prob(img)
            predd.append(nod_prob[0][1])
        if(len(predd)>0):
                predd.sort()
                predds.append(np.sum(predd[-100:])/100)
                print pat," ",np.sum(predd[-100:])/100

        else:
            predds.append(0.5)
            print pat," ",0.5

np.savetxt("theta1_DNN_40000.csv",nnet.theta1,delimiter=',',fmt='%10.10f')
np.savetxt("theta2_DNN_40000.csv",nnet.theta2,delimiter=',',fmt='%10.10f')
np.savetxt("theta3_DNN_40000.csv",nnet.theta3,delimiter=',',fmt='%10.10f')
np.savetxt("bias1_DNN_40000.csv",nnet.bias1,delimiter=',',fmt='%10.10f')
np.savetxt("bias2_DNN_40000.csv",nnet.bias2,delimiter=',',fmt='%10.10f')
np.savetxt("bias3_DNN_40000.csv",nnet.bias3,delimiter=',',fmt='%10.10f')

