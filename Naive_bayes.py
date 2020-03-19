import numpy as np
import math
class Naive_Bayesian:
    def __init__(self,num_train,num_test,num_feat):
        self.X_train=np.zeros((num_train,num_feat))
        self.Y_train=np.zeros(num_train)
        self.X_test=np.zeros((num_test,num_feat))
        self.Y_test=np.zeros(num_test)
        self.Y_predicted=np.zeros(num_test)
        self.num_train=num_train
        self.num_test=num_test
        self.num_feat=num_feat
    def normalization(self):
        mins=np.min(self.X_train,axis=0)
        maxs=np.max(self.X_train,axis=0)
        self.X_train=self.X_train/maxs
        maxs=np.max(self.X_test,axis=0)
        mins=np.min(self.X_test,axis=0)
        self.X_test=self.X_test/maxs
        #self.Y_train=np.where(self.Y_train==2,0,1)
        #self.Y_test=np.where(self.Y_test==2,0,1)
    def get_data(self):
        Data= np.genfromtxt(r'hepatitis.data.txt', delimiter=',')
        self.X_train=Data[:self.num_train,:self.num_feat]
        self.Y_train=Data[:self.num_train,self.num_feat]
        self.X_test=Data[self.num_train:,:self.num_feat]
        self.Y_test=Data[self.num_train:,self.num_feat]
        self.Y_train=np.where(self.Y_train==2,1,0)
        self.Y_test=np.where(self.Y_test==2,1,0)
        self.data_change()
        self.normalization()
    def normpdf(self,x, mean, sd):
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return float(num/denom)
    def class_splitting(self):
        X_train_1=[]
        X_train_0=[]
        #print(self.Y_train)
        for i in range(len(self.Y_train)):
            if(self.Y_train[i]==1):
                X_train_1.append(self.X_train[i])
            elif(self.Y_train[i]==0):
                X_train_0.append(self.X_train[i])
        self.X_train_1=np.asarray(X_train_1)
        self.X_train_0=np.asarray(X_train_0)
        #print(self.X_train_0)
    def conditional_densities(self):
        self.mean_0=np.sum(self.X_train_0,axis=0)/np.shape(self.X_train_0)[0]
        self.mean_1=np.sum(self.X_train_1,axis=0)/np.shape(self.X_train_1)[0]
        covariance_0=np.dot((self.X_train_0-self.mean_0).T,self.X_train_0-self.mean_0)/np.shape(self.X_train_0)[0]
        covariance_1=np.dot((self.X_train_1-self.mean_1).T,self.X_train_1-self.mean_1)/np.shape(self.X_train_1)[0]
        self.std_0=np.diagonal(covariance_0)
        self.std_1=np.diagonal(covariance_1)
        #print(self.mean_0,self.std_0)
    def train(self):
        self.get_data()
        self.class_splitting()
        self.conditional_densities()
    def test(self):
        l=0
        p_0=np.shape(self.X_train_0)[0]/np.shape(self.X_train)[0]
        for i in self.X_test:
            k=0
            x_0=[]
            x_1=[]
            for j in i:
                x_0.append(self.normpdf(j,self.mean_0[k],self.std_0[k]))
                x_1.append(self.normpdf(j,self.mean_1[k],self.std_1[k]))
                k=k+1
            fx0=np.prod(x_0)
            fx1=np.prod(x_1)
            #print(fx0,fx1)
            if(fx0*p_0>fx1*(1-p_0)):
                self.Y_predicted[l]=0
                l=l+1
            else:
                self.Y_predicted[l]=1
                l=l+1
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
def split_data():
    Data= np.genfromtxt(r'hepatitis.data.txt', delimiter=',')
    num_train=int(2*np.shape(Data)[0]/3)
    num_test=np.shape(Data)[0]-num_train
    num_feat=np.shape(Data)[1]-1
    return num_train,num_test,num_feat

num_train,num_test,num_feat=split_data()
nbay1=Naive_Bayesian(num_train,num_test,num_feat)
nbay1.train()
nbay1.test()
nbay1.accuracy()
