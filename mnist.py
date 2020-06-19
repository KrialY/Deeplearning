import torch
from torch.nn import init
from torch import nn
import numpy as np
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from IPython import display
from matplotlib import pyplot as plt
import time
import mymodule as Module

'''
对于不同的模型，不同的损失函数，不同的学习率，不同的优化模型等等，我们只需要修改对应参数即可。
'''

'''
这里的train_acc_784_300_100_10.pt中的数字代表不同模型参数的模型对应的存储文件
我们通过存储文件可以进行数据分析，也可以通过存储文件接着上次来训练
'''
try:
    train_acc_arr=torch.load('./data/train_acc_784_300_100_10.pt')
    test_acc_arr=torch.load('./data/test_acc_784_300_100_10.pt')
    lose_arr=torch.load('./data/lose_784_300_100_10.pt')
    test_lose_arr=torch.load('./data/testLose_784_300_100_10.pt')
    time_arr=torch.load('./data/time_784_300_100_10.pt')
except Exception:
    train_acc_arr=[]
    test_acc_arr=[]
    lose_arr=[]
    test_lose_arr=[]
    time_arr=[]

def load_data_mnist(batch_size=256):
    mnist_train = torchvision.datasets.MNIST(root='~/Datasets/', train=True, 
download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST(root='~/Datasets/', train=False, 
                                  download=True, transform=transforms.ToTensor())

    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, 
                                          shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, 
                                          shuffle=False, num_workers=num_workers)
    return train_iter,test_iter,mnist_train,mnist_test


    # 评估准确率
def evaluate_accuracy(data_iter, net,loss,getOnehotCode):
    l_sum,acc_sum, n = 0.0,0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            y_hat=net(X)
            m=getOnehotCode(y)
            l=loss(y_hat,m).sum()
            l_sum+=l.item()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            
    return l_sum/n,acc_sum / n
def evaluate_accuracy_without_onehot(data_iter, net,loss):
    l_sum,acc_sum, n = 0.0,0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            y_hat=net(X)
            l=loss(y_hat,y).sum()
            l_sum+=l.item()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            
    return l_sum/n,acc_sum / n
            
    return l_sum/n,acc_sum / n
def getOnehotCode(label):
    y=torch.zeros(label.size(0),10)
    for i,val in enumerate(label):
        y[i][label[i]]=1 
    return y
# 定义训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            m=getOnehotCode(y)
            l = loss(y_hat, m).sum()
            
            # 梯度清零
            optimizer.zero_grad()
            
            #计算梯度
            l.backward()
            
            # 使用梯度更新参数
            optimizer.step()  
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
#         test_loss,test_acc = evaluate_accuracy_without_onehot(test_iter, net,loss)
        test_loss,test_acc = evaluate_accuracy(test_iter, net,loss,getOnehotCode)
        train_acc_arr.append(train_acc_sum / n)
        test_acc_arr.append(test_acc)
        lose_arr.append(train_l_sum / n)
        test_lose_arr.append(test_loss)
        print('epoch %d, train_loss %f, test_loss %f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n,test_loss, train_acc_sum / n, test_acc))
        
# 1. 使用任务三的模型创建对象
net = Module.Activation_Net_2(28*28, 300, 10)
net2 = Module.Activation_Net_2(28*28, 1000, 10)
net3 = Module.Activation_Net_3(28*28, 300, 100,10)
net4 = Module.Activation_Net_3(28*28, 500, 150,10)
net5 = Module.Activation_Net_3(28*28, 500, 300,10)
net6 = Module.Activation_Net_2(28*28, 800, 10)

net=net3
for name,param in net.named_parameters():
    if 'weight' in name:
        init.kaiming_uniform_(param, nonlinearity='relu')
    if 'bias' in name:
        init.zeros_(param)

# 3. 选择损失函数
# myloss = torch.nn.CrossEntropyLoss() # 有的模型使用MSE
myloss=torch.nn.MSELoss()
#  4. 选择优化方法
optimizer = torch.optim.SGD(net.parameters(), lr=0.01 )
# optimizer = optim.Adam(net.parameters(), lr = 0.0001)

# 5. 加载数据并开始训练
batch_size = 256
train_iter,test_iter,mnist_train,mnist_test = load_data_mnist(batch_size)
epochs = 5

time1=time.time()

try:
    net.load_state_dict(torch.load('./data/net_784_300_100_10.pt'))
except Exception:
    print("No file exist,so do not init with file.")
    
train_ch3(net,train_iter,test_iter,myloss,epochs,batch_size, optimizer)
torch.save(net.state_dict(),'./data/net_784_300_100_10.pt')
time2=time.time()
time_arr.append(time2-time1)

torch.save(train_acc_arr,'./data/train_acc_784_300_100_10.pt')
torch.save(test_acc_arr,'./data/test_acc_784_300_100_10.pt')
torch.save(lose_arr,'./data/lose_784_300_100_10.pt')
torch.save(time_arr,'./data/time_784_300_100_10.pt')
torch.save(test_lose_arr,'./data/testLose_784_300_100_10.pt')


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')
# 本函数已保存在d2lzh包中方便以后使用
def show_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
    
a, b = [], []
for i in range(100):
    a.append(mnist_train[i][0])
    b.append(mnist_train[i][1])
show_mnist(a, b)#将图片显示出来