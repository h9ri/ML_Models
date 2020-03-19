import math
import numpy as np
class KNN:
    def __init__(self,k_near,num_train,num_test,num_feat):
        self.X_train=np.zeros((num_train,num_feat))
        self.Y_train=np.zeros(num_train)
        self.X_test=np.zeros((num_test,3))
        self.Y_test=np.zeros(num_test)
        self.k=k_near
        self.Y_predicted=np.zeros(num_test)
        self.distances=np.zeros((num_test,num_train))
        self.k_nearest=np.zeros((num_test,k_near))
        self.k_nearest_label=np.zeros((num_test,k_near))
        self.num_train=num_train
        self.num_test=num_test
        self.num_feat=num_feat
    def find_distances(self,x1,x2):
        return np.linalg.norm(x1 - x2);
    def normalization(self):
        mins=np.min(self.X_train,axis=0)
        maxs=np.max(self.X_train,axis=0)
        #self.X_train = 1 - ((maxs - self.X_train)/maxs-mins)
        self.X_train=self.X_train/maxs
        mins=np.min(self.X_test,axis=0)
        maxs=np.max(self.X_test,axis=0)
        #self.X_test = 1 - ((maxs - self.X_test)/maxs-mins)
        #self.X_train=(self.X_train-np.mean(self.X_train,axis=0))/np.std(self.X_train,axis=0)
        self.X_test=self.X_test/maxs
        #self.X_test=(self.X_test-np.mean(self.X_test))/np.std(self.X_test)
    def train(self):
        Data= np.genfromtxt(r'haberman.data.txt', delimiter=',')
        self.X_train=Data[:self.num_train,:self.num_feat]
        self.Y_train=Data[:self.num_train,self.num_feat]
        self.X_test=Data[self.num_train:,:self.num_feat]
        self.Y_test=Data[self.num_train:,self.num_feat]
        self.data_change()
        self.normalization()
    def test(self):
        l=m=0
        for i in self.X_test:
            m=0
            for j in self.X_train:
                self.distances[l][m]=self.find_distances(i,j);
                m=m+1
            l=l+1
        self.find_k_smallest()
        self.find_class()
    def find_k_smallest(self):
        j=l=m=0
        for i in self.distances:
            self.k_nearest[j]=np.argsort(i)[:self.k]
            j=j+1
        for i in self.k_nearest:
            m=0
            for j in i:
                j=j.astype(int)
                self.k_nearest_label[l][m]=self.Y_train[j]
                m=m+1
            l=l+1
    def find_class(self):
        j=0
        for i in self.k_nearest_label:
            i=i.astype(int)
            self.Y_predicted[j]=np.bincount(i).argmax()
            j=j+1
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
            if(self.Y_test[i]==2 and self.Y_predicted[i]==1):
                false_positive=false_positive+1
            if(self.Y_test[i]==1 and self.Y_predicted[i]==2):
                false_negitive=false_negitive+1
        precision=true_positive/(true_positive+false_positive)
        recall=true_positive/(true_positive+false_negitive)
        print("Accuracy:",accuracy/len(self.Y_test))
        print("Precision:",precision)
        print("Recall:",recall)
        print("F1:",2*precision*recall/(precision+recall))
def split_data():
    Data= np.genfromtxt(r'haberman.data.txt', delimiter=',')
    num_train=int(2*np.shape(Data)[0]/3)
    num_test=np.shape(Data)[0]-num_train
    num_feat=np.shape(Data)[1]-1
    return num_train,num_test,num_feat
num_train,num_test,num_feat=split_data()
knn1=KNN(5,num_train,num_test,num_feat)
knn1.train()
knn1.test()
knn1.accuracy()
