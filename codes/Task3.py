#!/usr/bin/env python
# coding: utf-8

# #### MNIST Image Classification

# In[1]:


import numpy as np
import pandas as pd
import glob
from matplotlib import image as img
import matplotlib.pyplot as plt
import random
import pickle as pk
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns


# #### Prepare data

# In[2]:


def loadDataset(path):
    print('Loading Dataset....')
    train_x,train_y,test_x,test_y = [],[],[],[]
    for i in range(10):
        for file in glob.glob(path+'/train/'+str(i)+'/*.png'):
            im = img.imread(file)
            train_x.append(im)
            train_y.append(i)
    for i in range(10):
        for file in glob.glob(path+'/test/'+str(i)+'/*.png'):
            im = img.imread(file)
            test_x.append(im)
            test_y.append(i)
    print('Dataset loaded....')

    return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)


# In[3]:


train_set_x, train_set_y, test_set_x, test_set_y = loadDataset ("C:/Users/Sumayya/Desktop/DeepLearning/Assignmnet1/MNIST_Data/MNIST_Data")
print(train_set_x.shape)
print(train_set_y.shape)
print(test_set_x.shape)
print(test_set_y.shape)


# In[4]:


img = train_set_x[26]
plt.savefig('train image.jpg')
plt.imshow(img)


# In[5]:


plt.imshow(test_set_x[1000])


# In[6]:


def mean_subtraction(train_set,test_set):
    all_imgs = np.vstack((train_set,test_set))
    sumd_img =  np.sum([i for i in all_imgs], 0)
    mean_img = sumd_img/all_imgs.shape[0]
    print('..Dataset Mean Image..')
    plt.imshow(mean_img)
    train_sub = train_set - mean_img
    test_sub = test_set - mean_img
    return train_sub,test_sub


# In[7]:


def vectorize(train_x,train_y,test_x,test_y):
    train_X = train_x.reshape((train_x.shape[0], train_x.shape[1]*train_x.shape[2]))
    train_Y = train_y.reshape((len(train_y),1))
    test_X = test_x.reshape((test_x.shape[0], test_x.shape[1]*test_x.shape[2]))
    test_Y = test_y.reshape((len(test_y),1))
    return train_X,train_Y,test_X,test_Y


# In[8]:


def shuffle_data(x,y):
    data = np.append(x,y,1)
    print(data.shape)
    np.random.shuffle(data)
    trainX = data[:,0:data.shape[1]-1]
    trainY = data[:,data.shape[1]-2:-1]
    print(trainX.shape,trainY.shape)
    return trainX,trainY


# In[9]:


def split_data(train_X,train_Y):
    #shuffle data
    trainX,trainY = shuffle_data(train_X,train_Y)
    
    #train chunk
    tr_size = int((trainX.shape[0]/3)*2.4)
    train_x = trainX[0:tr_size]
    train_y = trainY[0:tr_size]
    
    #valid chunk
    valid_x = trainX[tr_size:-1]
    valid_y = trainY[tr_size:-1]
    
    return train_x, train_y, valid_x, valid_y


# #### Mean Image Subtracted Data

# In[10]:


train_X_Sub, test_X_Sub = mean_subtraction(train_set_x,test_set_x)


# In[11]:


#vectorize the data set
m_train_X,train_Y,m_test_X,test_Y = vectorize(train_X_Sub, train_set_y, test_X_Sub, test_set_y )
print(m_train_X.shape)
print(train_Y.shape)
print(m_test_X.shape)
print(test_Y.shape)


# In[12]:


m_train_x, train_y, m_valid_x, valid_y = split_data(m_train_X,train_Y)
m_train_x.shape, train_y.shape, m_valid_x.shape, valid_y.shape


# #### Non Mean Image Subtracted Data

# In[13]:


#vectorize the data set
train_X,train_Y,test_X,test_Y = vectorize(train_set_x, train_set_y, test_set_x, test_set_y )
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)


# In[14]:


train_x, train_y, valid_x, valid_y = split_data(train_X,train_Y)
train_x.shape, train_y.shape, valid_x.shape, valid_y.shape


# #### Neural Network Class

# In[15]:


#class neural network
class MNISTClassification(object):
    def __init__(self,no_of_layers, input_dim, neurons_per_layer):
        #weights
        self.W1 = np.random.random((input_dim,neurons_per_layer[0])) # randomly initialize W1 using random function of numpy
        self.b1 = np.random.random((neurons_per_layer[0],1))
        
        # size of the wieght will be (inputSize +1, hiddenlayer) that +1 is for bias    
        self.W2 = np.random.random((neurons_per_layer[0],neurons_per_layer[1])) # randomly initialize W2 using random function of numpy
        self.b2 = np.random.random((neurons_per_layer[1],1))
        
        # size of the wieght will be (hiddenlayer +1, outputSize) that +1 is for bias    
        self.W3 = np.random.random((neurons_per_layer[1],neurons_per_layer[2])) # randomly initialize W2 using random function of numpy
        self.b3 = np.random.random((neurons_per_layer[2],1))
        #print(self.W1.shape,self.b1.shape,self.W2.shape,self.b2.shape,self.W3.shape,self.b3.shape)
        
        #arrays to store activatins of layers
        self.h2_act = []
        self.h1_act = []
        
        
    def miniBatches(self,inputX,b_size):
        mini_batches = []
        total_batches = int(inputX.shape[0]/b_size)
        count = 0
        for i in range(total_batches):
            mini = inputX[count:count+b_size]
            mini_batches.append(mini)
            count+=b_size
        return np.array(mini_batches)
      
    def oneHot(self,y):
        #y is a vector
        y_onehot = []
        for i in range(len(y)):
            onehot = np.zeros((10))
            onehot[int(y[i])] = 1
            y_onehot.append(onehot)
        y_onhot = np.array((y_onehot))
        return y_onhot
        
     
    def sigmoid(self,x):
        sig = 1/(1+np.exp(-x))
        return sig 
    
    
    def sigmoid_derivative(self, s):
        #derivative of sigmoid
        return s * (1-s)
    
    
    def softmax(self,x):
        arr = []
        X = x.T
        for i in X:
            z = np.exp(i-max(i))
            soft = np.exp(z) / np.sum(z)
            arr.append(soft)
        #e = np.exp(x)
        #soft = np.exp(e) / (np.sum(e, axis=0))
        arr = np.array((arr))
        arr = arr.reshape((x.shape))
        return arr
    
    
    def plotLossAcc(self,train,valid,title,operation):
        plt.figure(figsize=(7,5))
        plt.title("Train/Valid"+ operation)
        plt.plot(train,label="train")
        plt.plot(valid,label="validation")
        plt.xlabel("epochs")
        plt.ylabel(operation)
        plt.legend()
        plt.savefig(title)
        plt.show()
        
        
        
    def plotTSNE(self,dataX,dataY,label):
        tsne_plot = TSNE(n_components=2, verbose=1, random_state=123)
        x = tsne_plot.fit_transform(dataX)
        df = pd.DataFrame()
        df["Y"] = dataY
        df["dim1"] = x[:,0]
        df["dim2"] = x[:,1]
        #plot = sns.scatterplot(x="compA", y="compB", hue=df.Y.tolist(),
         #               palette=sns.color_palette("hls", 10),data=df).set(title=label)
        # Ploting the result of tsne
        sns.FacetGrid(df, hue="Y", size=6).map(plt.scatter, "dim1", "dim2").add_legend()
        plt.savefig(label+'jpg')
        plt.show()
    
    
    def accuracy(self, real,pred):
        correct_count = 0
        for y,y_hat in zip(real,pred):
            if np.argmax(y)==np.argmax(y_hat):
                correct_count+=1
        acc = (correct_count/real.shape[0]) *100
        return int(acc) 
    
   
   
    def softmaxToOnehot(self,y):
        pred = []
        #onehot encode Y
        for i in y.T:
            arr = np.zeros((y.shape[0]))
            arr[np.argmax(i)] = 1
            pred.append(arr)
        pred = np.array(pred).reshape((y.T.shape))
        return pred
    
     
    def feedForward(self,x):
        self.h2_act = []
        self.h1_act = []
        
        #transpose the x so that each column is a sample
        x = x.T
        
        #first layer
        a = np.dot(self.W1.T,x)+self.b1
        h1 = self.sigmoid(a)
        self.h1_act = h1
        
        #first hidden layer
        b = np.dot(self.W2.T,h1)+self.b2
        h2 = self.sigmoid(b)
        self.h2_act = h2
        
        #output  layer
        c = np.dot(self.W3.T,h2)+self.b3
        y = self.softmax(c)
        return y
        
        
        
    def crossEntropy(self, Y, Y_pred):
        # compute error based on crossentropy loss
        # where inputs are vectors
        n = Y.shape[1]
        loss = -(1/n) * (np.sum(np.dot(Y,np.log(Y_pred))))
        return loss
    
    
    
    def backProp(self,X, Y, y_pred, lr):
        y = Y.T #invert y to match shapes
        divisor = 1/Y.shape[1]
        
        #back prop w3 and b3
        #dl_dW3 = dl_dyhat * dyhat_dc * dc_dw3
        #dl_db3 = dl_dyhat * dyhat_dc * dc_db3
        dL_dc = y_pred-y
        dL_dW3 = divisor * np.dot(self.h2_act, dL_dc.T)
        dL_db3 = divisor * np.sum((y_pred-y), axis=1, keepdims=True)
        self.W3 = self.W3 - lr*dL_dW3
        self.b3 = self.b3 - lr*dL_db3
        
        #back prop w2 and b2, divide the products into chunks
        #dl_dW2 = dl_dc * dc_dh2 * dh2_db * db_dW2
        #dl_db2 = dl_dc * dc_dh2 * dh2_db * db_db2
        dc_dh2 = self.W3
        dh2_db = self.sigmoid_derivative(self.h2_act)
        db_dW2 = self.h1_act
        #compute it in chunks for debugging
        chunk1 = np.dot(dc_dh2,dL_dc)
        chunk2 = chunk1*dh2_db
        dL_dW2 = divisor* np.dot(db_dW2,chunk2.T)
        dL_db2 = divisor * np.sum(chunk2, axis=1, keepdims=True)
        self.W2 = self.W2 - lr*dL_dW2
        self.b2 = self.b2 - lr*dL_db2
        
        #back prop W1 and b1
        #dl_dW1 = dl_dc * dc_dh2 * dh2_db * db_dh1 * dh1_da * da_dw1
        #dl_db1 = dl_dc * dc_dh2 * dh2_db * db_dh1 * dh1_da * da_db1
        db_h1 = self.W2
        dh1_da = self.sigmoid_derivative(self.h1_act)
        da_dW1 = X
        #recalculate these chunks to get the effect of updated w3
        chunk1 = np.dot(dc_dh2,dL_dc)
        chunk2 = chunk1*dh2_db
        chunk3 = np.dot(db_h1,chunk2)
        chunk4 = chunk3 * dh1_da
        dL_dW1 = divisor* np.dot(da_dW1.T,chunk4.T)
        dL_db1 = divisor * np.sum(chunk4, axis=1, keepdims=True)
        self.W1 = self.W1 - lr*dL_dW1
        self.b1 = self.b1 - lr*dL_db1
    
    
   
    def trainMgd(self, train_x, train_y, valid_x, valid_y,learning_rate, batch_size, tr_epoch,plot_err=True,confusionM=True,tsne=True ):
        print("...Training...")
        
        #onehot encode train y and valid y
        tr_loss = []
        vl_loss = []
        tr_acc = []
        vl_acc = []
        trainY = self.oneHot(train_y)
        validY = self.oneHot(valid_y)
        
        #convert data into batches
        mini_trX = self.miniBatches(train_x,batch_size)
        mini_trY = self.miniBatches(trainY,batch_size)
        
        for i in range(tr_epoch):
            print("epoch: ",i)
            
            last_x = []
            last_y = []
            last_y_hat = []
            
            batch_loss = []
            batch_acc = []
            for x,y in zip(mini_trX,mini_trY):
                #forward pass
                y_pred = self.feedForward(x)
                
                #compute loss and accuracy for batch
                b_loss = self.crossEntropy(y,y_pred)
                batch_loss.append(b_loss)
                b_acc = self.accuracy(y,y_pred.T)
                batch_acc.append(b_acc)
                
                #backprop
                self.backProp(x,y,y_pred,learning_rate)
                
                #save the last batch data for the confusion matrix
                if i==tr_epoch-1:
                    last_x = x
                    last_y = y
                    last_y_hat = y_pred
                
            #add the average of batch loss to tr loss and accuracy
            avg_loss = "{:.2f}".format(np.average(np.array(batch_loss)) )
            tr_loss.append(avg_loss)
            avg_acc = "{:.2f}".format(np.average(np.array(batch_acc)))
            tr_acc.append(avg_acc) 
            print('Loss: ',avg_loss," ; Accuracy: ",avg_acc)
            
            #check validation
            valid_pred = self.feedForward(valid_x)
            vloss = self.crossEntropy( validY,valid_pred)
            v_acc = self.accuracy(validY,valid_pred.T)
            vl_loss.append(vloss) 
            vl_acc.append(v_acc)
            
            #plot confusion matrix and tsne for batch gradinat decent
            if confusionM and tsne and i==tr_epoch-1:
                prediction = np.argmax(last_y_hat, axis=0)
                #labels = mini_trY.reshape((y_pred.shape[1],y_pred.shape[0]))
                true_y = np.argmax(last_y.T, axis=0)
                print(confusion_matrix(true_y,prediction))
                
                #plot tsne for origonal data and the hidden layers 
                #dataX = mini_trX.reshape((mini_trX.shape[1],mini_trX.shape[2]))
                #dataY = mini_trY.reshape((mini_trY.shape[1],mini_trY.shape[2]))
                self.plotTSNE(last_x,true_y,"MNIST Classification Input Data")
                '''
                h1_out = np.argmax(self.h1_act, axis=0)
                #h1_out = self.softmaxToOnehot(self.h1_act)
                self.plotTSNE(dataX,h1_out,"MNIST Classification Hidden layer1")
                
                h2_out = np.argmax(self.h2_act, axis=0)
                #h2_out = self.softmaxToOnehot(self.h2_act)
                self.plotTSNE(dataX,h2_out,"MNIST Classification Hidden layer2")
                '''
                
        if plot_err:
            self.plotLossAcc(tr_loss,vl_loss,"Task#3 Loss.jpg"," Loss")
            self.plotLossAcc(tr_acc,vl_acc,"Task#3 accuracy.jpg"," Accuracy")
        
   

    def predict(self, testX):
        # predict the value of testX
        pred = self.feedForward(testX)
        return pred
    
    def saveModel(self,name):
        # save your trained model, it is your interpretation how, which and what data you store
        # which you will use later for prediction
        weights = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        with open(name+'.pkl', 'wb') as outfile:
            pk.dump(weights, outfile, pk.HIGHEST_PROTOCOL)



        
    def loadModel(self,name):
        # load your trained model, load exactly how you stored it.
        with open(name+'.pkl', 'rb') as infile:
             weights = pk.load(infile)


# #### Model instance with Mean Normalized Data

# #### Batch gradiant decent

# In[16]:


model = MNISTClassification(2,784,[128,64,10])
model.trainMgd(m_train_x, train_y, m_valid_x, valid_y,0.1,44763 , 1)
model.saveModel('batchGD')


# In[17]:


test_y_onehot = model.oneHot(test_Y)
test_pred = model.predict(m_test_X)
test_acc = model.accuracy(test_y_onehot,test_pred.T)
test_acc


# #### Mini Batch Gradiant Decent

# In[16]:


model = MNISTClassification(2,784,[128,64,10])
model.trainMgd(m_train_x, train_y, m_valid_x, valid_y,0.000001, 20000, 250)
model.saveModel('miniBatchGD')


# #### Predict test data

# In[18]:


test_y_onehot = model.oneHot(test_Y)
test_pred = model.predict(m_test_X)
test_acc = model.accuracy(test_y_onehot,test_pred.T)
test_acc


# In[ ]:


mm = MNISTClassification()
# load model which will be provided by you
mm.loadModel('bestmodel')
# check accuracy of that model
mm.accuracy(mtestX,testY)


# #### Model Instance without Mean Normalization

# In[16]:


model2 = MNISTClassification(2,784,[128,64,10])
#without mean image subtraction
model2.trainMgd(train_x, train_y, valid_x, valid_y,0.0000001, 20000, 10)
model2.saveModel('miniBatchGD_nonNormalized')


# In[ ]:




