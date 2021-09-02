import math
#from tsk.classifier import TSK
from sklearn.decomposition import PCA
from torch.nn import functional as F
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.optim import *
import math
import random
import numpy as np
from clustering import CMeansClustering
from torch.autograd import Variable
from sklearn import preprocessing
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans
from torch.utils.data import random_split
import time

#device = torch.device( 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False
print(torch.cuda.is_available())
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)
length = 9
num_class = 7
batch_size = 50
num_epochs = 1200
pm = 0.65
learning_rate = 0.001
NumFuzz = 10
NumRule = 8
NumEnhan = 10
rng = np.random.default_rng()
def get_random_list(start,stop,n):
    '''
    生成范围在[start,stop], 长度为n的数组.
    区间包含左右endpoint
    '''
    arr = list(range(start, stop+1))
    shuffle_n(arr,n)
    return arr[-n:]

def shuffle_n(arr,n):

    random.seed(time.time())
    for i in range(len(arr)-1,len(arr)-n-1,-1):
        
        j = random.randint(0,i)
        arr[i], arr[j] = arr[j], arr[i]
lst = get_random_list(0,79,24)
data = np.loadtxt(open("zoo.csv", "rb"), delimiter = ",", skiprows = 0)


data = rng.permutation(data)
data[:,1:] = preprocessing.scale(data[:,1:])

d = np.array_split(data,(80,))

#build label noised data
for k in lst:
    l = random.randint(0,num_class-1)
    while(d[0][k][0]==l):
        l = random.randint(0,num_class-1)
    d[0][k][0]= l
    #d[0][k][0] = not d[0][k][0]

Center=[]
for i in range(NumFuzz):
    kmeans = KMeans(n_clusters=NumRule,init='random',n_init=1).fit(d[0][:,1:])
    centers = kmeans.cluster_centers_
    Center.append(centers)
Center = np.array(Center)

input_dim = d[0][:,1:].shape[1]
Alpha = np.random.rand(NumFuzz,input_dim*NumRule,NumRule)

WeightEnhan = np.random.rand(NumFuzz*NumRule+1,NumEnhan)

def compute_firing_level(data: np.ndarray, centers: int, delta: float) -> np.ndarray:
    """
    Compute firing strength using Gaussian model

    :param data: n_Samples * n_Features
    :param centers: data center，n_Clusters * n_Features
    :param delta: variance of each feature， n_Clusters * n_Features
    :return: firing strength
    """

    d = -(np.expand_dims(data, axis=2) - np.expand_dims(centers.T, axis=0)) ** 2 / (2 * delta)
    
    d = np.exp(np.sum(d, axis=1))
    
    d = np.fmax(d, np.finfo(np.float64).eps)

    return d / np.sum(d, axis=1, keepdims=True)


def apply_firing_level_o(x: np.ndarray, firing_levels: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Convert raw input to tsk input, based on the provided firing levels
    :param x: (np.ndarray) Raw input
    :param firing_levels: (np.ndarray) Firing level for each rule
    :param order: (int) TSK order. Valid values are 0 and 1
    :return:
    """
    if order == 0:
        return firing_levels
    else:
        n = x.shape[0]
        firing_levels = np.expand_dims(firing_levels, axis=1)
        x = np.expand_dims(x, axis=2)
        x = np.repeat(x, repeats=firing_levels.shape[2], axis=2)

        output = x * firing_levels

        

        output = output.reshape([n, -1])
        

        
        return output
    
def normalize(data):
    Dmax,Dmin = data.max(axis=0),data.min(axis=0)
    data = (data-Dmin)/(Dmax-Dmin)
    return data

def FBLS_pre(data,numfuzz,numrule,alpha,weightEnhan):
    y = np.zeros((data.shape[0],numfuzz*numrule))
    
    for i in range(numfuzz):
        a = Alpha[i]
        mu_a = compute_firing_level(data, Center[i], 1)
        d = apply_firing_level_o(data, mu_a, 1)
        T1 = d @ a
        T1 = normalize(T1)
        
        y[:,numrule*(i):numrule*(i+1)] = T1
    H = np.concatenate((y, np.ones([y.shape[0], 1])), axis=1)
    T = H @ weightEnhan
    l = np.max(T)
    l1 = 0.8/l
    T = np.tanh(T*l1)
    T2 = np.concatenate((y, T), axis=1)
    return T2

def update_ema_variables(model, ema_model,alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
                                        
class myDataset(Dataset):
    def __init__(self,data_set,root_dir,NumFuzz,NumRule,Alpha,WeightEnhan):
        #data_set = np.loadtxt(open(path, "rb"), delimiter = ",", skiprows = 0)
        self.data = data_set[:,1:]
        

        self.data = FBLS_pre(self.data,NumFuzz,NumRule,Alpha,WeightEnhan)

        self.label = data_set[:,0]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        data = (self.data[idx],self.label[idx])
        return data
    
#data_set = myDataset("haberman.csv","D:/python35/2020RAtrain",NumFuzz,NumRule,Alpha,WeightEnhan)
#train_dataset,test_dataset = random_split(data_set, [240,66])
train_dataset = myDataset(d[0],"D:/python35/2020RAtrain",NumFuzz,NumRule,Alpha,WeightEnhan)

#train_dataset,train_no_dataset = random_split(train_dataset, [len(train_dataset)-int(len(train_dataset)*p),int(len(train_dataset)*p)])
print(len(train_dataset))
#train_no_dataset = myDataset("e_train_no.csv","D:/python35/2020RAtrain",center,variance)
test_dataset = myDataset(d[1],"D:/python35/2020RAtrain",NumFuzz,NumRule,Alpha,WeightEnhan)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=len(train_dataset), 
                                           shuffle=True)
#train_no_loader = torch.utils.data.DataLoader(dataset=train_no_dataset,
#                                          batch_size=len(train_no_dataset), 
#                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=len(test_dataset), 
                                          shuffle=False)

class NN(nn.Module):
    def __init__(self):

        super(NN,self).__init__()

        self.fc1 = nn.Linear(NumRule*NumFuzz+NumEnhan,num_class)

        #self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.fc1(x)
        
        return x
def softmax_mse_loss(input_logits, target_logits):

    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes
    
class TSLoss(torch.nn.Module):
 
     def __init__(self):
         super(TSLoss, self).__init__()

 
     def forward(self, output1,output2,label1):

         #loss1 = F.mse_loss(output1_w, label1,reduction='mean')
         loss1 = F.cross_entropy(output1, label1)
         loss2 = softmax_mse_loss(output1, output2) #F.mse_loss(output1_n, output2_n,reduction='mean')
         
         return loss1+loss2
        
model_student = NN()
model_teacher = NN()
#criterion = nn.MSELoss(reduction='mean')#nn.CrossEntropyLoss()#
#criterion =nn.CrossEntropyLoss()
criterion = TSLoss()
#criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.Adam(model_student.parameters(), lr=learning_rate)

total_step = len(train_loader)

w=None
total = 0
correct = 0
for data, labels in train_loader:
    data = data.numpy()
    labels = labels.long()
    total += labels.size(0)
    l = labels.reshape(len(train_dataset), 1)
    l = l.long()
    y = torch.zeros(len(train_dataset),num_class).scatter_(1,l,1)
    print(data.shape)
    
    w = np.linalg.solve((np.conj(data.T)@data+np.identity(data.shape[1])*2**-30),np.conj(data.T)@y.numpy())
    o = data@w
    o = torch.from_numpy(o)
    _, predicted = torch.max(o.data, 1)
    print(predicted.shape)
    correct += (predicted == labels).sum().item()
    print(100 * correct / total)
total = 0
correct = 0
for data, labels in test_loader:
    data = data.numpy()
    labels = labels.long()
    o = data@w
    o = torch.from_numpy(o)
    _, predicted = torch.max(o.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print(100 * correct / total)
'''
for epoch in range(num_epochs):
    #for (data1, labels1),(data2,labels2) in zip(train_loader,train_no_loader):
    for data1, labels1 in train_loader:
        #labels = labels.reshape(len(train_dataset), 1)
        labels1 = labels1.long()
        
        #y = torch.zeros(len(train_dataset),num_class).scatter_(1,labels,1)
        outputs1 = model_student(data1)
        
        #outputs = model_student(data2)
        outputs2 = model_teacher(data1)
        
        #loss_t = criterion(outputs1,labels)
        loss_t = criterion(outputs1,outputs2,labels1)
        alpha = min(1 - 1 / (epoch + 1), pm)
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        update_ema_variables(model_student,model_teacher,alpha)
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},alpha:{}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss_t.item(),alpha))



model_student.eval()   
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in train_loader:

        labels = labels.long()
        outputs = model_student(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the student_model on the 10000 train images: {} %'.format(100 * correct / total))
                                     
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in train_loader:

        labels = labels.long()
        outputs = model_teacher(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the teacher_model on the 10000 train images: {} %'.format(100 * correct / total))        

with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:

        labels = labels.long()
        outputs = model_student(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    print('Test Accuracy of the student model on the 10000 test images: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        #data = data.reshape(input_dim,-1 , 1)
        #data = data.reshape(-1, 1500)
        labels = labels.long()
        outputs = model_teacher(data)
        #outputs = outputs.reshape(-1,num_class)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the teacher model on the 10000 test images: {} %'.format(100 * correct / total)) 

#torch.save(model.state_dict(), 'model.ckpt')
'''
