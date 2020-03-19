import numpy as np
import math
import random
class Logistic_Regression:
    def __init__(self,epoch,learning_rate,num_train,num_test,num_feat):
        self.num_epoch=epoch
        self.alpha=learning_rate
        self.X_train=np.zeros((num_train,num_feat))
        self.Y_train=np.zeros(num_train)
        self.X_test=np.zeros((num_test,num_feat))
        self.Y_test=np.zeros(num_test)
        self.weight=np.zeros(num_feat)
        self.biases=np.zeros(num_train)
        self.num_train=num_train
        self.num_test=num_test
        self.num_feat=num_feat
    def normalization(self):
        mins = np.min(self.X_train, axis = 0)
        maxs = np.max(self.X_train, axis = 0)
        #self.X_train = self.X_train-mins/(maxs-mins)
        self.X_train = self.X_train/maxs
        maxs=np.max(self.X_test,axis=0)
        mins=np.min(self.X_test,axis=0)
        #self.X_test = self.X_test-mins/(maxs-mins)
        self.X_test=self.X_test/maxs
    def data_change(self):
        sum_array=[]
        y=self.X_train.T
        for i in y:
            count=0
            sum=0
            for j in i:
                if(math.isnan(j)):
                    continue
                else:
                    count=count+1
                    sum=sum+j
            sum_array.append(sum/count)
        for i in range(np.shape(y)[0]):
            for j in range(np.shape(y)[1]):
                if(math.isnan(y[i][j])):
                    y[i][j]=sum_array[i]
        self.X_train=y.T
        y=self.X_test.T
        for i in range(np.shape(y)[0]):
            for j in range(np.shape(y)[1]):
                if(math.isnan(y[i][j])):
                    y[i][j]=sum_array[i]
        self.X_test=y.T
    def conversion(self):
        self.Y_train=np.where(self.Y_train==2,1,0)
        self.Y_train=self.Y_train.astype(int)
        self.Y_test=np.where(self.Y_test==2,1,0)
        self.Y_test=self.Y_test.astype(int)
    def get_data(self):
        Data= np.genfromtxt(r'hepatitis.data.txt', delimiter=',')
        self.X_train=Data[:self.num_train,1:]
        self.Y_train=Data[:self.num_train,0]
        self.X_test=Data[self.num_train:,1:]
        self.Y_test=Data[self.num_train:,0]
        self.data_change()
        self.normalization()
        self.conversion()
        self.initialization()
    def initialization(self):
        self.W=.01*np.random.randn(np.shape(self.X_train)[1],)
        #self.biases=0.1*np.random.randn(np.shape(self.X_train)[0],)
    def train(self):
        self.get_data()
        for i in range(self.num_epoch):
            #print(self.sigmoid(self.X_train,self.W))
            self.W=self.W-(self.alpha*np.dot((self.sigmoid(self.X_train,self.W)-self.Y_train),self.X_train))#/np.shape(self.X_train)[0]
            #print(self.cost_function())
        #print(self.W)
    def sigmoid(self,X,W):
        return (1.0/(1 + np.exp(-np.dot(X, W))))
    def cost_function(self):
        log_func = self.sigmoid(self.X_train,self.W)
        step1 = self.Y_train * np.log(log_func)
        step2 = (1 - self.Y_train) * np.log(1 - log_func)
        final = -step1 - step2
        return np.mean(final)/np.shape(self.X_train)[0]
    def loss_function(self):
        log_func = self.sigmoid(self.X_train,self.W)
        return np.mean(np.square(log_func-Y_train)/np.shape(self.X_train)[0])
    def pred_values(self):
        pred_prob = self.sigmoid(self.X_train,self.W)
        pred_value = np.where(pred_prob > .50, 1, 0)
        count=0
        for i in range(np.shape(self.Y_train)[0]):
            if self.Y_train[i]==pred_value[i]:
                count+=1
    def test(self):
        pred_prob = self.sigmoid(self.X_test,self.W)
        pred_value = np.where(pred_prob > .50, 1, 0)
        self.Y_predicted=pred_value
    def accuracy(self):
        accuracy=0
        true_positive=0
        false_positive=0
        false_negitive=0
        print(self.Y_predicted)
        for i in range(len(self.Y_test)):
            if(self.Y_test[i]==self.Y_predicted[i]):
                accuracy=accuracy+1
            if(self.Y_test[i]==1 and self.Y_predicted[i]==1):
                true_positive=true_positive+1
            if(self.Y_test[i]==0 and self.Y_predicted[i]==1):
                false_positive=false_positive+1
            if(self.Y_test[i]==1 and self.Y_predicted[i]==0):
                false_negitive=false_negitive+1
        precision=true_positive/(true_positive+false_positive)
        recall=true_positive/(true_positive+false_negitive)
        print("Accuracy:",accuracy/len(self.Y_test))
        print("Precision:",precision)
        print("Recall:",recall)
        print("F1:",2*precision*recall/(precision+recall))
def split_data():
    Data= np.genfromtxt(r'hepatitis.data.txt', delimiter=',')
    print(np.shape(Data))
    num_train=int(2*np.shape(Data)[0]/3)
    num_test=np.shape(Data)[0]-num_train
    print(np.shape(Data))
    num_feat=np.shape(Data)[1]-1
    return num_train,num_test,num_feat
num_train,num_test,num_feat=split_data()
lr1=Logistic_Regression(5000,1,num_train,num_test,num_feat)
lr1.train()
#lr1.pred_values()
lr1.test()
lr1.accuracy()
