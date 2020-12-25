import os
import dataloader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import time
'''
linear1=torch.nn.Linear(N_FEATURES, hiddenLayerSize, bias=True)
torch.nn.init.xavier_uniform_(linear1.weight)
'''

class DeepConvNet(nn.Module):
    def __init__(self,func_name):
        super(DeepConvNet, self).__init__()
        def activation_function(func_name=''):
            if (func_name == 'RELU'):
                return nn.ReLU()
            elif (func_name == 'Leaky_RELU'):
                return nn.LeakyReLU()
            else:
                return nn.ELU(alpha=1.0)
        A1 = nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1))
        A2 = nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1))
        A3 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        A4 = activation_function()
        A5 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
        A6 = nn.Dropout(p=0.5)
        self.modelA = nn.Sequential(A1,A2,A3,A4,A5,A6)

        B1 = nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1))
        B2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        B3 = activation_function()
        B4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
        B5 = nn.Dropout(p=0.5)
        self.modelB = nn.Sequential(B1,B2,B3,B4,B5)

        C1 = nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1))
        C2 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        C3 = activation_function()
        C4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
        C5 = nn.Dropout(p=0.5)
        self.modelC = nn.Sequential(C1,C2,C3,C4,C5)

        D1 = nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1))
        D2 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        D3 = activation_function()
        D4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)
        D5 = nn.Dropout(p=0.5)
        self.modelD = nn.Sequential(D1,D2,D3,D4,D5)

        self.modelE = nn.Linear(in_features=8600, out_features=2, bias=True)

    def forward(self, x):

        x = self.modelA(x)
        #print('after model A', x.shape)
        x = self.modelB(x)
        #print('after model B', x.shape)
        x = self.modelC(x)
        #print('after model C', x.shape)
        x = self.modelD(x)
        #print('after model D', x.shape)
        x = x.view(-1, 8600)
        x = self.modelE(x)
        #print('after model E', x.shape)

        return x

    def save(self, path, epoch, accuracy):
        state = {
            'state': self.state_dict(),
            'epoch': epoch                   # 将epoch一并保存
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(path, 'checkpoint', 'DeepConvNet_'+str(accuracy)))

    def load(self, weight_file_path):
        checkpoint = torch.load(weight_file_path)
        self.load_state_dict(checkpoint['state'])        # 从字典中依次读取
        start_epoch = checkpoint['epoch']
        return start_epoch

class EGG(torch.nn.Module):
    def __init__(self, func_name):
        super(EGG, self).__init__()
        def activation_function(func_name=''):
            if (func_name == 'RELU'):
                return nn.ReLU()
            elif (func_name == 'Leaky_RELU'):
                return nn.LeakyReLU()
            else:
                return nn.ELU(alpha=1.0)
        # A: firstconv
        # B: depthwiseConv
        # C: separableConv
        # D: classify
        self.func_name = func_name

        A1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        A2 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #torch.nn.init.xavier_uniform_(A1.weight)
        #torch.nn.init.xavier_uniform_(A2.weight)
        self.modelA = nn.Sequential(A1,A2)

        B1 = nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)
        B2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        B3 = activation_function(self.func_name)
        B4 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        B5 = nn.Dropout(p=0.25)
        #torch.nn.init.xavier_uniform_(B1.weight)
        #torch.nn.init.xavier_uniform_(B2.weight)
        #torch.nn.init.xavier_uniform_(B4.weight)
        self.modelB = nn.Sequential(B1,B2,B3,B4,B5)

        C1 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        C2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        C3 = activation_function (self.func_name)
        C4 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        C5 = nn.Dropout(p=0.25)
        #torch.nn.init.xavier_uniform_(C1.weight)
        #torch.nn.init.xavier_uniform_(C2.weight)
        #torch.nn.init.xavier_uniform_(C4.weight)

        self.modelC = nn.Sequential(C1,C2,C3,C4,C5)

        self.out = nn.Linear(in_features=736, out_features=2, bias=True)

    def forward(self, x):
        #print('Init', x.shape)
        x = self.modelA(x)
        #print('after model A', x.shape)
        x = self.modelB(x)
        #print('after model B', x.shape)
        x = self.modelC(x)
        #print('after model C', x.shape)
        x = x.view(x.shape[0],736)
        x = self.out(x)
        #print('out x',x.shape)
        return x

    def save(self, path, epoch, accuracy):
        state = {
            'state': self.state_dict(),
            'epoch': epoch                   # 将epoch一并保存
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(path, 'checkpoint', self.func_name+'_'+str(accuracy)))

    def load(self, weight_file_path):
        checkpoint = torch.load(weight_file_path)
        self.load_state_dict(checkpoint['state'])        # 从字典中依次读取
        start_epoch = checkpoint['epoch']
        return start_epoch

def test(test_loader,comment):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: %f %%' % (100 * correct / total),comment)
    return (100.0 * correct) / total



MAX_EPOCH = 100
LR = 0.01

#path = 'G:/VM_SYNC/Deeplearning/lab2'
#os.chdir('G:/VM_SYNC/Deeplearning/lab2')

device = torch.device('cpu')
a,b,c,d = dataloader.read_bci_data()

train_x = torch.tensor(a,dtype=torch.float,device=device)
train_y = torch.tensor(b,dtype=torch.long,device=device)

test_x = torch.tensor(c,dtype=torch.float,device=device)
test_y = torch.tensor(d,dtype=torch.long,device=device)
test_dataset = Data.TensorDataset(test_x,test_y)
test_loader = Data.DataLoader(
    dataset = test_dataset,
    batch_size=1080,
    shuffle=False,
    num_workers=0
)

#netA = DeepConvNet('ELU')
#netB = EGG('ELU')
#input()

ACCURACYS = []
for fn in ['RELU','Leaky_RELU', 'ELU']:
#for fn in ['ELU']:
    ACCUARCY_TRAIN = []
    ACCUARCY_TEST = []
    #net = EGG()
    net = DeepConvNet(fn)
    net = net.float()
    #net.load(os.path.join(path,'checkpoint','Leaky_RELU_86.57'))
    #net.eval()
    #test(test_loader)
    optimizer = optim.Adam(net.parameters() ,lr=LR)
    criterion = nn.CrossEntropyLoss()
    max_test_accuracy = 70
    for BATCH_SIZE in [64,64,64]:
        train_dataset = Data.TensorDataset(train_x,train_y)
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        for epoch in range(MAX_EPOCH):
            for step, (batch_x, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()   # zero the gradient buffers
                output = net(batch_x)
                output = output.float()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()    # Does the update
                #print('epoch: ', epoch, '| step: ', step, '| loss', loss.item())

            net.eval()
            train_accuracy = test(train_loader,fn+'train')
            ACCUARCY_TRAIN.append(train_accuracy)
            test_accuracy = test(test_loader,fn+'test')
            ACCUARCY_TEST.append(test_accuracy)

            if test_accuracy > max_test_accuracy:
                net.save(path,epoch,round(test_accuracy, 2))
                max_test_accuracy = test_accuracy

            #end = time.time()
            #print('#################################')
            #print('epoch ', epoch, 'use time', end-start, "seconds.")
            #print('#################################')
            #start = end

    ACCURACYS.append(ACCUARCY_TRAIN)
    ACCURACYS.append(ACCUARCY_TEST)

count = 0
colors = ['red', 'yellow', 'blue', 'green', 'skyblue', 'brown']
x_axis= np.arange(0,MAX_EPOCH*3,1)
lines = []
for fn in ['RELU','Leaky_RELU', 'ELU']:
    A, = plt.plot(x_axis, ACCURACYS[count], color= colors[count], label= fn+'training accuracy ')
    B, = plt.plot(x_axis, ACCURACYS[count+1], color= colors[count+1], label=fn+'testing accuracy ')
    lines.append(A)
    lines.append(B)
    count += 2


plt.legend(handles=lines,labels=['RELU_train','RELU_test','Leaky_RELU_train','Leaky_test','ELU_train','ELU_test'],loc='best')
plt.savefig('G:/VM_SYNC/Deeplearning/lab2/DCN_compare.png')



'''
EEGNet
(firstconv): Sequential(
(O): Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False) (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running stats=True)
(depthwiseConv): Sequential(
(0 : Conv2d (16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False) (1): BatchNorm2d (32, eps=1e-05, momentum=0.1, affine=True, track_running stats=True) (2): ELU (alpha=1.0) (3): AvgPool2d (kernel_size=(1, 4), stride=(1, 4), padding=0); (4): Dropout(p=0.25)
(separableConv): Sequential(
(0): Conv2D(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False) (1): BatchNorm2d (32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) (2): ELU (alpha=1.0) (3): AvgPool2d (kernel_size=(1, 8), stride=(1, 8), padding=0); (4): Dropout(p=0.25)
(classify): Sequential
(0): Linear(in features=736, out_features=2, bias=True)
'''