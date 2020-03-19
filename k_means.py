import numpy as np
import math
Data= np.genfromtxt(r'hepatitis.data.txt', delimiter=',')
X=Data[:,1:]
Y=Data[:,0]
k=2
label=np.zeros((np.shape(X)[0]))
iter=500
def find_distances(x1,x2):
    return np.linalg.norm(x1 - x2);
sum_array=[]
y=X.T
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
X=y.T
X=X/np.max(X,axis=0)
Y=np.where(Y==2,0,1)
c0=X[1]
c1=X[6]
while(iter>0):
    j=0
    cluster0=[]
    cluster1=[]
    for i in X:
        distances=[]
        distances.append(find_distances(i,c0))
        distances.append(find_distances(i,c1))
        label[j]=np.argmin(distances)
        if(label[j]==0):
            cluster0.append(i)
        else:
            cluster1.append(i)
        j=j+1
    c0=np.mean(cluster0,axis=0)
    c1=np.mean(cluster1,axis=0)
    iter=iter-1
#label=np.where(label==1,0,1)
accuracy=0
true_positive=0
false_positive=0
false_negitive=0
print(label)
for i in range(len(Y)):
    if(Y[i]==label[i]):
        accuracy=accuracy+1
    if(Y[i]==1 and label[i]==1):
        true_positive=true_positive+1
    if(Y[i]==0 and label[i]==1):
        false_positive=false_positive+1
    if(Y[i]==1 and label[i]==0):
        false_negitive=false_negitive+1
precision=true_positive/(true_positive+false_positive)
recall=true_positive/(true_positive+false_negitive)
print("Accuracy:",accuracy/len(Y))
print("Precision:",precision)
print("Recall:",recall)
print("F1:",2*precision*recall/(precision+recall))
