import numpy as np
# --im2col--方便正向传播时图像的卷积计算
def im2col(images,flt_h,flt_w,out_h,out_w,stride,pad):
    n_bt,n_ch,img_h,img_w=images.shape
    img_pad=np.pad(images,[(0,0),(0,0),(pad,pad),(pad,pad)],"constant")
    cols=np.zeros((n_bt,n_ch,flt_h,flt_w,out_h,out_w))
    for h in range(flt_h):
        h_him=h+stride*out_h
        for w in range(flt_w):
            w_him=w+stride*out_w
            cols[:,:,h,w,:,:]=img_pad[:,:,h:h_him:stride,w:w_him:stride]
    cols=cols.transpose(1,2,3,0,4,5).reshape(n_ch*flt_h*flt_w,n_bt*out_h*out_w)
    return cols
#--col2im--用于反向传播时将梯度的cols转换为输入时的图像尺寸大小
def col2im(cols,img_shape,flt_h,flt_w,out_h,out_w,stride,pad):
    n_bt,n_ch,img_h,img_w=img_shape
    cols=cols.reshape(n_ch,flt_h,flt_w,n_bt,out_h,out_w).transpose(3,0,1,2,4,5)
    images=np.zeros((n_bt,n_ch,img_h+2*pad+stride-1,img_w+2*pad+stride-1))
    for h in range(flt_h):
         h_him=h+stride*out_h
         for w in range(flt_w):
             w_him=w+stride*out_w
             images[:,:,h:h_him:stride,w:w_him:stride]+=cols[:,:,h,w,:,:]
    return  images[:,:,pad:img_h+pad,pad:img_w+pad]
#--卷积层--
class ConvLayer:
    #n_bt:批次尺寸，x_ch：输入的通道数量，x_h:输入时图像的高度，x_w:输入图像的宽度
    #n_fit：过滤器的数量，flt_h:过滤器的高度，flt_w:过滤器的宽度
    #stride：步长幅度，pad：填充的幅度
    #y_ch：输出的通道数量，y_h：输出的高度，y_w:输出的宽度


    def __init__(self,x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad):

        self.params=(x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad)
        #过滤器和偏置的初始值
        self.w=wb_width*np.random.randn(n_flt,x_ch,flt_h,flt_w)
        self.b=wb_width*np.random.randn(1,n_flt)
        #输出的通道数，长和宽
        self.y_ch=n_flt
        self.y_h=(x_h-flt_h+2*pad)//stride+1
        self.y_w=(x_w-flt_w+2*pad)//stride+1
    def forward (self,x):
        n_bt=x.shape[0]
        x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad=self.params
        y_ch,y_h,y_w=self.y_ch,self.y_h,self.y_w
        self.cols=im2col(x,flt_h,flt_w,y_h,y_w,stride,pad)
        self.w_col=self.w.reshape(n_flt,x_ch*flt_h*flt_w)
        u=np.dot(self.w_col,self.cols).T+self.b
        self.u=u.reshape(n_bt,y_h,y_w,y_ch).transpose(0,3,1,2)
        self.y=np.where(self.u<=0,0,self.u)
    def backward(self,grad_y):
        n_bt=grad_y.shape[0]
        x_ch,x_h,x_w,n_flt,flt_h,flt_w,stride,pad=self.params
        y_ch,y_h,y_w=self.y_ch,self.y_h,self.y_w
        delta=grad_y*np.where(self.u<=0,0,1)
        delta=delta.transpose(0,2,3,1).reshape(n_bt*y_h*y_w,y_ch)
        grad_w=np.dot(self.cols,delta)
        self.grad_w=grad_w.T.reshape(n_flt,x_ch,flt_h,flt_w)
        self.grad_b=np.sum(delta,axis=0)
        grad_cols=np.dot(delta,self.w_col)
        x_shape=(n_bt,x_ch,x_h,x_w)
        self.grad_x=col2im(grad_cols.T,x_shape,flt_h,flt_w,y_h,y_w,stride,pad)
class   PoolingLayer:
    def __init__(self,x_ch,x_h,x_w,pool,pad):
        self.params=(x_ch,x_h,x_w,pool,pad)
        self.y_ch=x_ch
        self.y_h=x_h//pool if x_h%pool==0 else x_h//pool+1
        self.y_w=x_w//pool if x_w%pool==0 else x_w//pool+1
    def forward(self,x):
        n_bt=x.shape[0]
        x_ch,x_h,x_w,pool,pad=self.params
        y_ch,y_h,y_w=self.y_ch,self.y_h,self.y_w
        cols=im2col(x,pool,pool,y_h,y_w,pool,pad)
        cols=cols.T.reshape(n_bt*y_h*y_w*x_ch,pool*pool)
        y=np.max(cols,axis=1)
        self.y=y.reshape(n_bt,y_h,y_w,x_ch).transpose(0,3,1,2)
        self.max_index=np.argmax(cols,axis=1)
    def backward(self,grad_y):
        n_bt=grad_y.shape[0]
        x_ch,x_h,x_w,pool,pad=self.params
        y_ch,y_h,y_w=self.y_ch,self.y_h,self.y_w
        grad_y=grad_y.transpose(0,2,3,1)
        grad_cols=np.zeros((pool*pool,grad_y.size))
        grad_cols[self.max_index.reshape(-1), np.arange(grad_y.size)]=grad_y.reshape(-1)
        grad_cols=grad_cols.reshape(pool,pool,n_bt,y_h,y_w,y_ch)
        grad_cols=grad_cols.transpose(5,0,1,2,3,4)
        grad_cols=grad_cols.reshape(y_ch*pool*pool,n_bt*y_h*y_w)
        x_shape=(n_bt,x_ch,x_h,x_w)
        self.grad_x=col2im(grad_cols,x_shape,pool,pool,y_h,y_w,pool,pad)
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
        self.y=np.where(self.u<=0,0,self.u)
    def backward(self,grad_y):
        delta=grad_y*np.where(self.u<=0,0,1)
        self.grad_w=np.dot(self.x.T,delta)
        self.grad_b=np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)
class OutputLayer(BaseLayer):
    def forward(self,x):
        self.x=x
        u=np.dot(x,self.w)+self.b
        self.y=np.exp(u)/np.sum(np.exp(u),axis=1).reshape(-1,1)
    def backward(self,t):
        delta=self.y-t
        self.grad_w=np.dot(self.x.T,delta)

        self.grad_b=np.sum(delta,axis=0)
        self.grad_x=np.dot(delta,self.w.T)
import matplotlib.pyplot as plt
from sklearn import datasets
#手写文字的数据集读入
digits_data=datasets.load_digits()
input_data=digits_data.data
plt.imshow(input_data[0].reshape(8,8),cmap="gray")
plt.show()
correct=digits_data.target
n_data=len(correct)
#数据标准化
ave_input=np.average(input_data)
std_input=np.std(input_data)
inputa_data=(input_data-ave_input)/std_input
plt.imshow(inputa_data[0].reshape(8,8),cmap="gray")
plt.show()
#将答案转化为独热编码格式
correct_data=np.zeros((n_data,10))
for i in range(n_data):
    correct_data[i,correct[i]]=1
#训练数据和测试数据
index=np.arange(n_data)
index_train=index[index%3!=0]
index_test=index[index%3==0]
input_train=input_data[index_train,:]
correct_train=correct_data[index_train,:]
input_test=input_data[index_test,:]
correct_test=correct_data[index_test,:]

n_train=input_train.shape[0]
n_test=input_test.shape[0]
#--各个设置值--
img_h=8
img_w=8
img_ch=1
wb_width=0.1
eta=0.01
epoch=50
batch_size=8
interval=10
n_sample=200












