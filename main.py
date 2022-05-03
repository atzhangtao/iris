import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
###数据的读入
iris_data=datasets.load_iris()
input_data=iris_data.data
correct=iris_data.target
n_data=len(input_data)


#input数据的z标准化
u=np.average(input_data,axis=0)
std=np.std(input_data,axis=0)
input_data=(input_data-u)/std
#out结果转化为独热码
correct_data=np.zeros((n_data,3))
for i in range(n_data):
    correct_data[i,correct[i]]=1
#划分训练数据和测试数据
index=np.arange(n_data)
index_train=index[index%2==0]
index_test=index[index%2!=0]
input_train=input_data[index_train,:]
input_test=input_data[index_test,:]
correct_train=correct_data[index_train,:]
correct_test=correct_data[index_test,:]
n_train=input_train.shape[0]
n_test=input_test.shape[0]
# 设置超参数
wb_width=0.1
n_in=4
n_mid=50
n_out=3
eta=0.01
epoch=1000
batch_size=8
interval=100


#手动编写middle和output结构
class BaseLayer():
    def __init__(self,n_upper,n):
        self.w=wb_width*np.random.randn(n_upper,n)
        self.b=wb_width*np.random.randn(n)
        self.h_w=np.zeros((n_upper,n))+1e-8
        self.h_b=np.zeros(n)+1e-8
    def updata(self,eta):
        self.h_w+=self.grad_w*self.grad_w
        self.h_b+=self.grad_b*self.grad_b
        self.w-=eta/np.sqrt(self.h_w)*self.grad_w
        self.b-=eta/np.sqrt(self.h_b)*self.grad_b

class MiddleLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        self.u=np.dot(x,self.w)+self.b
        self.y=np.where(self.u>0,self.u,0)

    def backford(self,grad_y):
        delta=grad_y*np.where(self.u<=0,0,1)
        self.grad_w=np.dot(self.x.T,delta)
        self.grad_b=np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)
class OutputLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        self.u=np.dot(x,self.w)+self.b
        self.y=np.exp(self.u)/np.sum(np.exp(self.u),axis=1,keepdims=True)
    def backford(self,t):
        delta=self.y-t
        self.grad_w=np.dot(self.x.T,delta)
        self.grad_b=np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)
class Dropout(BaseLayer):
    def __init__(self,dropout_ratio):
           self.drapout_ratio=dropout_ratio
    def forward(self,x,is_train):
        if is_train:
            rand=np.random.rand(*x.shape)
            self.dropout=np.where(rand>=self.drapout_ratio,1,0)
            self.y=self.dropout*x
        else:
            self.y=(1-self.drapout_ratio)*x

    def backford(self,grad_y):
        self.grad_x=grad_y*self.dropout


#初始化层
middle_layer_1=MiddleLayer(n_in,n_mid)
dropout_1=Dropout(0.5)
middle_layer_2=MiddleLayer(n_mid,n_mid)
dropout_2=Dropout(0.5)
output_layer=OutputLayer(n_mid,n_out)
#正向传播

def forward_propagation(x,is_train):
    middle_layer_1.forward(x)
    dropout_1.forward(middle_layer_1.y,is_train)
    middle_layer_2.forward(dropout_1.y)
    dropout_2.forward(middle_layer_2.y,is_train)
    output_layer.forward(dropout_2.y)
#反向传播
def backford_propagatiom(t):
    output_layer.backford(t)
    dropout_2.backford(output_layer.grad_x)

    middle_layer_2.backford(dropout_2.grad_x)
    dropout_1.backford(middle_layer_2.grad_x)
    middle_layer_1.backford(dropout_1
                            .grad_x)

#权重和偏置的更新
def updata_wb():
    middle_layer_1.updata(eta)
    middle_layer_2.updata(eta)
    output_layer.updata(eta)

#编写误差函数

def get_err(t,batch_size):
    return -np.sum(t*np.log(output_layer.y+1e-7))/batch_size

# 用于记录误差
train_x_err=[]
train_y_err=[]
test_x_err=[]
test_y_err=[]
#--开始训练（epoch）--
for i in range(epoch):
#记录每个epoch的测试集和训练集的误差
  forward_propagation(input_train,False)
  error_train=get_err(correct_train,n_train)
  forward_propagation(input_test,False)
  error_test=get_err(correct_test,n_test)
  train_x_err.append(i)
  train_y_err.append(error_train)
  test_x_err.append(i)
  test_y_err.append(error_test)
#显示训练进度
  if i%interval==0:
     print("Epoch"+str(i)+"/"+str(epoch),
           "Error_train"+str(error_train),
           "Error_test"+str(error_test))



#设置随机训练坐标
  index_random=np.arange(n_train)
  np.random.shuffle(index_random)


#--开始学习--
  for j in range(n_train//batch_size):
#当前批次的训练数据
    md_index=index_random[(j)*batch_size:(j+1)*batch_size]
    x=input_train[md_index,:]
    t=correct_train[md_index,:]

#正向传播
    forward_propagation(x,True)
#反向传播
    backford_propagatiom(t)
#更新参数
    updata_wb()
#--结束一epoch学习--
#输出误差图像
  if(i%interval==0):
   plt.plot(train_x_err,train_y_err,label="Train")
   plt.plot(test_x_err,test_y_err,label="Test")
   plt.legend()
   plt.xlabel("epoch")
   plt.ylabel("Err")
   plt.show()
   plt.close()

#--结束训练-
#计算正确率
forward_propagation(input_train,False)
count_train=np.sum(np.argmax(output_layer.y,axis=1)==np.argmax(correct_train,axis=1))
forward_propagation(input_test,False)
count_test=np.sum(np.argmax(output_layer.y,axis=1)==np.argmax(correct_test,axis=1))
print("ACC train"+str(count_train/n_train*100)+"%",
      "ACC test"+str(count_test/n_test*100)+"%")
