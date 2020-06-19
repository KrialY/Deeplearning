import torch
from matplotlib import pyplot as plt

train_acc_arr=torch.load('./data/train_acc_784_500_150_10.pt')
test_acc_arr=torch.load('./data/test_acc_784_500_150_10.pt')
lose_arr=torch.load('./data/lose_784_500_150_10.pt')
test_lose_arr=torch.load('./data/testLose_784_500_150_10.pt')
times=torch.load('./data/time_784_500_150_10.pt')
time=0
for i in times:
    time=time+i
print((time/len(lose_arr))*50)
print(lose_arr[49])
print(test_lose_arr[49])
print(train_acc_arr[49])
print(test_acc_arr[49])

x=[]
for i,val in enumerate(lose_arr):
    x.append(i)
    
import matplotlib.pyplot as plt


#绘制网格
plt.figure(figsize=(20,8),dpi=80)
plt.grid(alpha=0.4,linestyle=':')
# plt.figure(figsize=(20,8),dpi=80)
plt.ylim((0, 0.000025))
plt.title('Lose')
plt.plot(x,lose_arr,label="train_lose")
plt.plot(x,test_lose_arr,label="test_lose")
plt.xlabel('epochs')
plt.ylabel('lose')
#添加图例
plt.legend(loc="upper left")
plt.show()


plt.figure(figsize=(20,8),dpi=80)
plt.grid(alpha=0.4,linestyle=':')
# plt.xlim((1100, 1110))
plt.ylim((0.975, 1))
plt.title('Train accuracy&Test accuracy')
plt.plot(x,train_acc_arr,label="train_acc")
plt.plot(x,test_acc_arr,label="test_acc")
plt.xlabel('epochs')
plt.ylabel('accuracy')
#添加图例
plt.legend(loc="upper left")
plt.show()