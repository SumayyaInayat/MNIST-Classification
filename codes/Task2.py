#!/usr/bin/env python
# coding: utf-8

# In[1]:


from asyncio.windows_events import NULL
import numpy as np
import random
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import pickle as pk


# In[2]:


data, label = ds.make_circles(n_samples=1000, factor=.4, noise=0.05)
#Lets visualize the dataset
reds = label == 0
blues = label == 1
plt.scatter(data[reds, 0], data[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(data[blues, 0], data[blues, 1], c="blue", s=20, edgecolor='k')
plt.show()


# In[3]:


random.Random(4).shuffle(data)
random.Random(4).shuffle(label)
#Note: shuffle this dataset before dividing it into three parts
# Distribute this data into three parts i.e. training, validation and testing

trainX = data[0:700]# training data point
trainY = label[0:700]# training lables
trainY = trainY.reshape((1,len(trainY)))

validX = data[700:800] # validation data point
validY = label[700:800]# validation lables
validY = validY.reshape((1,len(validY)))

testX = data[800:1000]# testing data point
testY = label[800:1000]# testing lables
testY = testY.reshape((1,len(testY)))


# In[14]:


class Neural_Network(object):        
    def __init__(self,inputSize = 2,hiddenlayer = 3, outputSize = 1 ):        
        # size of layers
        self.inputSize = inputSize
        self.outputSize = outputSize 
        self.hiddenLayer = hiddenlayer
        #weights
        self.W1 = np.random.random((inputSize+1,hiddenlayer)) # randomly initialize W1 using random function of numpy
        # size of the wieght will be (inputSize +1, hiddenlayer) that +1 is for bias    
        self.W2 = np.random.random((hiddenlayer+1,outputSize)) # randomly initialize W2 using random function of numpy
        # size of the wieght will be (hiddenlayer +1, outputSize) that +1 is for bias  
        self.L1_4_act = []
        self.L1_3_act =[]
        self.L2_act = []
   
    def sigmoid(self, s):
        # activation function
        sig = 1/(1+np.exp(-s))
        return sig # apply sigmoid function on s and return it's value
    
    def sigmoid_derivative(self, s):
        #derivative of sigmoid
        return s * (1-s) 
    
    def tanh(self, s):
        # activation function
        return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s)) 
    
    def tanh_derivative(self, s):
        #derivative of tanh
        return 1-s**2 
    
    def relu(self, s):
        # activation function
        if s<=0:
            return 0
        else:
            return 1
        
        
    def relu_derivative(self, s):
        #derivative of relu
        if s<=0:
            return 0
        else: 
            return 1 
        
        
    def crossentropy(self, Y, Y_pred):
        # compute error based on crossentropy loss
        # where inputs are vectors
        n = Y.shape[1]
        loss = -(1/n) * (np.sum(Y*np.log(Y_pred)) + np.sum((1-Y)*np.log(1-Y_pred)))
        return loss
   
    def plot_loss(self,trLoss,vLoss):
        plt.figure(figsize=(7,5))
        plt.title("Train/Valid Loss")
        plt.plot(trLoss,label="train")
        plt.plot(vLoss,label="validation")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('Task2_losses')
        plt.show()
    
    def feedforward(self, input_data):
        #arrays to hold activations for back prop
        self.L1_4_act = []
        self.L1_3_act =[]
        self.L2_act = []
        
        #invert input data so that every column is a sample
        input_data = input_data.T
        X = np.append(input_data, np.ones((1,input_data.shape[1])), axis=0)
        z = np.dot(self.W1.T,X)
        h = self.sigmoid(z)
        self.L1_3_act = h
        h = np.append(h, np.ones((1,h.shape[1])), axis=0)
        self.L1_4_act = h
        o = np.dot(self.W2.T,h)
        y_hat = self.sigmoid(o)
        self.L2_act = y_hat
        #y = y_hat.reshape((input_data.shape[1],))
        #y_hat = [1 if i >0.5 else 0 for i in y]
        #y_hat = np.array((y_hat)).reshape((1,input_data.shape[1]))
        return y_hat
    
    def backwardpropagate(self,X, Y, y_pred, lr):
        #dL_dW2 = dL_yPred * dyPred_dO * dO_dW2
        #append 1 to the shape of X and reshape it
        X = X.T
        X = np.append(X, np.ones((1,X.shape[1])), axis=0)
        divisor = 1/X.shape[1]
        dL_dW2 = divisor * np.dot(self.L1_4_act,(y_pred - Y).T)
        self.W2 = self.W2 - lr*dL_dW2
        
        #update W1
        #dL_dW1 = dL_yPred * dyPred_dO * dO_dH * dH_dZ * dZ_dW1
        #dL_yPred * dyPred_dO = (y_pred-Y).T
        dL_dO = y_pred - Y
        dO_dH = self.W2
        dO_dH = np.delete(dO_dH, 0, 0) #delete a row from W2 to match the dims
        dH_dZ = self.sigmoid_derivative(self.L1_3_act)
        
        #compute it in chunks for debugging
        chunk1 = np.dot(dO_dH,dL_dO)
        chunk2 = chunk1*dH_dZ
        dL_dW1 = divisor* np.dot(chunk2,X.T)
        self.W1 = self.W1 - lr*dL_dW1
    
    def train(self, trainX, trainY, epochs, learningRate = 0.001, plot_err = True ,validationX = NULL, validationY = NULL):
        print('...Trainig...')
        train_loss = []
        valid_loss = []
        for i in range(epochs):
            # feed forward trainX and trainY and recivce predicted value
            y_pred = self.feedforward(trainX)
            
            #calculate the loss and append it
            loss = self.crossentropy( trainY, y_pred)
            train_loss.append(loss)
            
            #backpropagate through the network
            self.backwardpropagate(trainX, trainY, y_pred, learningRate)
            # if validationX and validationY are not null than show validation accuracy and error of the model by printing values.
            if validationX is not NULL:
                valid_pred = self.feedforward(validationX)
                vloss = self.crossentropy( validationY,valid_pred)
                valid_loss.append(vloss)
            
        
            acc = self.accuracy(trainY,y_pred)
            print('Epoch ',i,';; Accuracy:',"{:.2f}".format(acc), ';; Loss:',"{:.2f}".format(loss))
        # plot error of the model if plot_err is true
        if plot_err:
            self.plot_loss(train_loss,valid_loss)
            
    def predict(self, testX):
        # predict the value of testX
        pred = self.feedforward(testX)
        return pred

    
    def accuracy(self, true, pred):
        y = pred.reshape((pred.shape[1],))
        pred = np.array(([1 if i >0.5 else 0 for i in y]))
        true = true.reshape((true.shape[1],))
        correct_count = 0
        for i in range(len(true)):
            if true[i] == pred[i]:
                correct_count+=1
        acc = correct_count/len(true) *100
        return acc 
    

    def saveModel(self,name):
        # save your trained model, it is your interpretation how, which and what data you store
        # which you will use later for prediction
        weights = [self.W1,self.W2]
        with open(name+'.pkl', 'wb') as outfile:
            pk.dump(weights, outfile, pk.HIGHEST_PROTOCOL)



        
    def loadModel(self,name):
        # load your trained model, load exactly how you stored it.
        with open(name+'.pkl', 'rb') as infile:
             weights = pk.load(infile)


# In[15]:


model = Neural_Network(2,3,1)
model.train(trainX, trainY, 250, 0.01, plot_err = True,validationX = validX, validationY = validY)


# In[16]:


# try different combinations of epochs and learning rate

#save the best model which you have trained, 
model.saveModel('task2model')
# create class object
mm = Neural_Network()
# load model which will be provided by you
mm.loadModel('task2model')
# check accuracy of that model
pred = mm.predict(testX)
mm.accuracy(testY,pred)


# In[17]:


plt.bar(["Train","Test"], [51.29,47.0])
plt.title('Test Train Accuracy')
plt.xlabel('Data')
plt.ylabel('Accuracy')
plt.savefig('Accuracy Task2.jpg')
plt.show()


# In[ ]:




