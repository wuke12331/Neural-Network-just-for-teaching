import numpy as np
from numba import vectorize
# Size 为列表,为神经网络结构，比如[3，5，5，4，2]，3是输入层神经元个数，中间为隐藏层每层神经元个数，2为输出层个数

class nn_Creat():
    def __init__(self,Size,active_fun='sigmoid',learning_rate=1.5,batch_normalization=1,objective_fun='MSE',
                 output_function='sigmoid',optimization_method='normal',weight_decay=0):

        self.Size=Size                                     # 初始化网络参数，并进行打印
        print('the structure of the NN is \n', self.Size)
        self.active_fun=active_fun
        print('active function is %s '% active_fun)
        self.learning_rate=learning_rate
        print('learning_rate is %s '% learning_rate)
        self.batch_normalization=batch_normalization
        print('batch_normalization is %d '% batch_normalization)
        self.objective_fun=objective_fun
        print('objective_function is %s '% objective_fun)
        self.optimization_method=optimization_method
        print('optimization_method is %s '% optimization_method)
        self.weight_decay = weight_decay
        print('weight_decay is %f '% weight_decay)

        # 初始化网络权值和梯度
        self.vecNum=0
        self.depth=len(Size)
        self.W=[]
        self.b=[]
        self.W_grad=[]
        self.b_grad=[]
        self.cost=[]
        if self.batch_normalization:    # 是否运用批量归一化，如果用，则引入期望E和方差S，以及缩放因子Gamma、Beta
            self.E = []
            self.S = []
            self.Gamma = []
            self.Beta = []

        if objective_fun=='Cross Entropy':  # 目标函数是否为交叉墒函数
            self.output_function='softmax'
        else:
            self.output_function='sigmoid'
        print('output_function is %s \n'% self.output_function)
        print('Start training NN \n')

        for item in range(self.depth-1):
            width=self.Size[item]
            height=self.Size[item+1]
            q=2*np.random.rand(height,width)/np.sqrt(width)-1/np.sqrt(width)    #初始化权系数W

            self.W.append(q)

            if  self.active_fun=='relu': # 判断激活函数是否为relu函数，以决定b的初始化形式
                self.b.append(np.random.rand(height,1)+0.01)
            else:
                self.b.append(2*np.random.rand(height,1)/np.sqrt(width)-1/np.sqrt(width))

            if self.optimization_method=='Momentum':  #优化方向是否使用矩形式，即为之前梯度的叠加
               if item!=0:
                    self.vW.append(np.zeros([height,width]))
                    self.vb.append(np.zeros([height, 1]))
               else:
                    self.vW=[]
                    self.vb=[]
                    self.vW.append(np.zeros([height, width]))
                    self.vb.append(np.zeros([height, 1]))

            if self.optimization_method=='AdaGrad'or optimization_method=='RMSProp' or optimization_method=='Adam': #优化方法是否使用上述方法
               if item!=0:
                    self.rW.append(np.zeros([height,width]))
                    self.rb.append(np.zeros([height, 1]))
               else:
                    self.rW=[]
                    self.rb=[]
                    self.rW.append(np.zeros([height, width]))
                    self.rb.append(np.zeros([height, 1]))

            if self.optimization_method == 'Adam':  #优化方法是否为Adam方法
                if item!=0:
                    self.sW.append(np.zeros([height, width]))
                    self.sb.append(np.zeros([height, 1]))
                else:
                    self.sW = []
                    self.sb = []
                    self.sW.append(np.zeros([height, width]))
                    self.sb.append(np.zeros([height, 1]))

            if self.batch_normalization:         #是否对每层进行归一化
                self.Gamma.append(np.array([1]))
                self.Beta.append(np.array([0]))
                self.E.append(np.zeros([height,1]))
                self.S.append(np.zeros([height,1]))
                if self.optimization_method=='Momentum':      #在归一化基础上是否使用Momentun方法
                    if item!=0:
                        self.vGamma.append(np.array([1]))
                        self.vBeta.append(np.array([0]))
                    else:
                        self.vGamma = []
                        self.vBeta = []
                        self.vGamma.append(np.array([1]))
                        self.vBeta.append(np.array([0]))

                if self.optimization_method == 'AdaGrad' or optimization_method == 'RMSProp' or optimization_method == 'Adam':  # 在归一化基础上优化方法是否使用上述方法
                    if item!=0:
                        self.rGamma.append(np.array([0]))
                        self.rBeta.append(np.array([0]))
                    else:
                        self.rGamma = []
                        self.rBeta = []
                        self.rGamma.append(np.array([0]))
                        self.rBeta.append(np.array([0]))

                if self.optimization_method == 'Adam':   #在归一化基础上是否使用Adam方法
                    if item!=0:
                        self.sGamma.append(np.array([1]))
                        self.sBeta.append(np.array([0]))
                    else:
                        self.sGamma = []
                        self.sBeta = []
                        self.sGamma.append(np.array([1]))
                        self.sBeta.append(np.array([0]))

            self.W_grad.append(np.array([]))
            self.b_grad.append(np.array([]))

    def nn_train(self,train_x,train_y,iterations=10,batch_size=100):  #神经网络训练流程化
        # 随机将数据分为num_batches堆，每堆Batch_Size个
        Batch_Size=batch_size
        m=np.size(train_x,0)
        num_batches=np.round(m/Batch_Size)
        num_batches=np.int(num_batches)
        for k in range(iterations):
            kk=np.random.randint(0,m,m)
            for l in range(num_batches):
                batch_x=train_x[kk[l*batch_size:(l+1)*batch_size  ],:]
                batch_y=train_y[kk[l*batch_size:(l+1)*batch_size  ],:]
                self.nn_forward(batch_x,batch_y)                            # 执行神经网络向前传播
                self.nn_backward(batch_y)                                   # 执行神经网络向后传播
                self.gradient_obtain()                                      # 执行得到所以参数的梯度

        return None

    def Sigmoid(self,z):     # 定义sigmoid函数
        yyy=1/(1+np.exp(-z))
        return yyy

    def SoftMax(self,x):       # 定义Softmax函数
        e_x = np.exp(x - np.max(x,0))
        return e_x / np.sum(e_x,0)

    def Relu(self,xxx):  # 定义Relu函数
        # xxx[xxx<0]=0
        s=np.maximum(xxx, 0)
        return s

    def nn_forward(self,batch_x,batch_y):   # 神经网络向前传播，得到对z偏导theta，每层输出a和cost

        batch_x=batch_x.T
        batch_y=batch_y.T
        m=np.size(batch_x,1)
        self.a=[]                               #定义每层激活函数输出
        self.a.append(batch_x)                  # 接受第一层输入
        cost2=0                                 #初始化正则函数
        self.yy=[]

        for k in range(1,self.depth):          # 从第一层开始，循环求每层输出
            y=(self.W[k-1].dot(self.a[k-1]))+(np.repeat(self.b[k-1],m,1))

            if  self.batch_normalization:
                self.E[k-1]=self.E[k-1]*self.vecNum+np.sum(y,1)[:,None]
                self.S[k-1]=self.S[k-1]**2*(self.vecNum-1)+((m-1)*np.std(y,1)**2)[:,None]
                self.vecNum=self.vecNum+m
                self.E[k-1]=self.E[k-1]/self.vecNum               #求期望
                self.S[k-1]=np.sqrt(self.S[k-1]/(self.vecNum-1))  #求方差

                y = (y - self.E[k-1]) / (self.S[k-1] + 0.0001)
                self.yy.append(y)                                 #存储缩放之前输入，以便反向传播使用
                y=self.Gamma[k-1]*y+self.Beta[k-1]                #缩放

                # 定义输出和激活函数字典，分别对应输出层和隐藏层

            if k==self.depth-1:                               #将每层输出函数值矩阵添加到相应数据列表
                target_output_function = {'sigmoid': self.Sigmoid, 'tanh': np.tanh,
                                          'relu': self.Relu, 'softmax': self.SoftMax}

                self.a.append(target_output_function[self.output_function](y))
            else:
                target_active_function = {'sigmoid': self.Sigmoid, 'tanh': np.tanh,
                                          'relu': self.Relu}

                self.a.append(target_active_function[self.active_fun](y))
                cost2=cost2+np.sum(self.W[k-1]**2)               #得到正则化损失函数

        # 得到总损失函数
        if self.objective_fun == 'MSE':
            self.cost.append(0.5 / m * np.sum((self.a[-1] - batch_y) ** 2) / m + 0.5 * self.weight_decay * cost2)
        elif self.objective_fun == 'Cross Entropy':
            self.cost.append(-0.5 * np.sum(batch_y * np.log(self.a[-1])) / m + 0.5 * self.weight_decay * cost2)


        return None

    # 神经网络反响传播得到参数梯度
    def nn_backward(self,batch_y):
        batch_y=batch_y.T
        m=np.size(self.a[0],1)
        # 不同输出函数损失函数梯度字典

        self.theta=[np.array([]) for i in range(self.depth)]                      # 初始化theta

        if   self.output_function=='sigmoid':
             self.theta[-1]=-(batch_y-self.a[-1] )*self.a[-1]*(1-self.a[-1])
        elif self.output_function=='tanh':
             self.theta[-1] = -(batch_y-self.a[-1] )*(1-self.a[-1]**2)
        elif self.output_function=='softmax':
             self.theta[-1] =self.a[-1]-batch_y

        if self.batch_normalization:
            self.gamma_grad = [np.array([]) for ii in range(self.depth - 1)]       # 若使用归一化，则初始化gamma和beta的梯度
            self.beta_grad = [np.array([]) for iii in range(self.depth - 1)]
            temp=self.theta[-1]*self.yy[-1]
            self.gamma_grad[-1]=np.sum(np.mean(temp,1))
            self.beta_grad[-1]=np.sum(np.mean(self.theta[-1],1))
            self.theta[-1]=self.Gamma[-1]*(self.theta[-1])/(self.S[-1]+0.0001)       #得到最后一个theta

        self.W_grad[-1]=self.theta[-1].dot(self.a[-2].T)/m+self.weight_decay*self.W[-1]                     #得到参数W和b的最后一个梯度向量
        self.b_grad[-1]=(np.sum(self.theta[-1],1)/m)[:,None]

        # 由最后一层参数，反向逐层求参数梯度
        for k in range(2,self.depth):

            if  self.active_fun=='sigmoid':
                self.theta[-k] = (self.W[-k + 1].T.dot(self.theta[-k + 1])) * (self.a[-k] * (1 - self.a[-k]))
            elif self.active_fun=='tanh':
                self.theta[-k] = (self.W[-k + 1].T.dot(self.theta[-k + 1])) * (1 - self.a[-k] ** 2)
            elif self.active_fun=='relu':
                self.theta[-k] = (self.W[-k + 1].T.dot(self.theta[-k + 1])) * (self.a[-k]>=0)


            if self.batch_normalization:
                temp=self.theta[-k]*self.yy[-k]
                self.gamma_grad[-k]=np.sum(np.mean(temp,1))
                self.beta_grad[-k]=np.sum(np.mean(self.theta[-k],1))
                self.theta[-k]=self.Gamma[-k]*(self.theta[-k])/((self.S[-k]+0.0001).reshape(np.size(self.S[-k]),1))
            self.W_grad[-k]=self.theta[-k].dot(self.a[-k-1].T)/m+self.weight_decay*self.W[-k]
            self.b_grad[-k]=(np.sum(self.theta[-k],1)/m)[:,None]



    def gradient_obtain(self):
    # 获取参数梯度信息
        for k in range(self.depth-1):

            if self.batch_normalization==0:
                if self.optimization_method=='normal':
                    self.W[k]= self.W[k]-self.learning_rate*self.W_grad[k]
                    self.b[k]=self.b[k]-self.learning_rate*self.b_grad[k]

                elif self.optimization_method=='AdaGrad':
                    self.rW[k]=self.rW[k]+self.W_grad[k]**2
                    self.rb[k]=self.rb[k]+self.b_grad[k]**2
                    self.W[k]=self.W[k]-self.learning_rate*self.W_grad[k]/(np.sqrt(self.rW[k])+0.001)
                    self.b[k]=self.b[k]-self.learning_rate*self.b_grad[k]/(np.sqrt(self.rb[k])+0.001)

                elif self.optimization_method=='Momentum':
                    rho=0.1
                    self.vW[k]=rho*self.vW[k]-self.learning_rate*self.W_grad[k]
                    self.vb[k] = rho * self.vb[k] - self.learning_rate * self.b_grad[k]
                    self.W[k]=self.W[k]+self.vW[k]
                    self.b[k]=self.b[k]+self.vb[k]
                elif self.optimization_method=='RMSProp':
                    rho=0.9
                    self.rW[k] = rho * self.rW[k] + 0.1* self.W_grad[k]**2
                    self.rb[k] = rho * self.rb[k] +0.1 * self.b_grad[k]**2
                    self.W[k] = self.W[k] - self.learning_rate*self.W_grad[k]/(np.sqrt(self.rW[k])+0.001)
                    self.b[k] = self.b[k] - self.learning_rate*self.b_grad[k]/(np.sqrt(self.rb[k])+0.001)

                elif self.optimization_method=='Adam':
                    rho1=0.9
                    rho2=0.999
                    self.sW[k]=0.9*self.sW[k]+0.1*self.W_grad[k]
                    self.sb[k]=0.9*self.sb[k]+0.1*self.b_grad[k]
                    self.rW[k]=0.999*self.rW[k]+0.001*self.W_grad[k]**2
                    self.rb[k]=0.999*self.rb[k]+0.001*self.b_grad[k]**2

                    newS=self.sW[k]/(1-rho1)
                    newR=self.rW[k]/(1-rho2)
                    self.W[k]=self.W[k]-self.learning_rate*newS/np.sqrt(newR+0.00001)
                    newS = self.sb[k] / (1 - rho1)
                    newR = self.rb[k] / (1 - rho2)
                    self.b[k]=self.b[k]-self.learning_rate*newS/np.sqrt(newR+0.00001)

            else:
                if self.optimization_method=='normal':
                    self.W[k]=self.W[k]-self.learning_rate*self.W_grad[k]
                    self.b[k]=self.b[k]-self.learning_rate*self.b_grad[k]
                    self.Gamma[k]=self.Gamma[k]-self.learning_rate*self.gamma_grad[k]
                    self.Beta[k]=self.Beta[k]-self.learning_rate*self.beta_grad[k]

                elif self.optimization_method=='AdaGrad':
                    self.rW[k]=self.rW[k]+self.W_grad[k]**2
                    self.rb[k]=self.rb[k]+self.b_grad[k]**2

                    self.rGamma[k]=self.rGamma[k]+self.gamma_grad[k]**2
                    self.rBeta[k]=self.rBeta[k]+self.beta_grad[k]**2

                    self.W[k]=self.W[k]-self.learning_rate*self.W_grad[k]/(np.sqrt(self.rW[k])+0.001)
                    self.b[k]=self.b[k]-self.learning_rate*self.b_grad[k]/(np.sqrt(self.rb[k])+0.001)

                    self.Gamma[k]=self.Gamma[k] - self.learning_rate * self.gamma_grad[k]/(
                                np.sqrt(self.rGamma[k])+0.001)
                    self.Beta[k] = self.Beta[k] - self.learning_rate * self.beta_grad[k] /(
                                np.sqrt(self.rBeta[k]) + 0.001)

                elif self.optimization_method=='Momentum':
                    rho=0.1
                    self.vW[k]=rho*self.vW[k]-self.learning_rate*self.W_grad[k]
                    self.vb[k] = rho * self.vb[k] - self.learning_rate * self.b_grad[k]
                    self.vGamma[k]=rho*self.vGamma[k]-self.learning_rate*self.gamma_grad[k]
                    self.vBeta[k]=rho*self.vBeta[k]-self.learning_rate*self.beta_grad[k]

                    self.W[k]=self.W[k]+self.vW[k]
                    self.b[k]=self.b[k]+self.vb[k]
                    self.Gamma[k]=self.Gamma[k]+self.vGamma[k]
                    self.Beta[k]-self.Beta[k]+self.vBeta[k]

                elif self.optimization_method=='RMSProp':
                    self.rW[k] = 0.9 * self.rW[k] + 0.1* self.W_grad[k]**2
                    self.rb[k] = 0.9 * self.rb[k] + 0.1 * self.b_grad[k]**2
                    self.rGamma[k]=0.9*self.rGamma[k]+0.1*self.gamma_grad[k]**2
                    self.rBeta[k]=0.9*self.rBeta[k]+0.1*self.beta_grad[k]**2

                    self.W[k] = self.W[k] - self.learning_rate*self.W_grad[k]/(np.sqrt(self.rW[k])+0.001)
                    self.b[k] = self.b[k] - self.learning_rate*self.b_grad[k]/(np.sqrt(self.rb[k])+0.001)
                    self.Gamma[k]=self.Gamma[k]-self.learning_rate*self.gamma_grad[k]/(
                                np.sqrt(self.rGamma[k])+0.001)
                    self.Beta[k] = self.Beta[k] - self.learning_rate * self.beta_grad[k] / (
                                np.sqrt(self.rBeta[k]) + 0.001)

                elif self.optimization_method=='Adam':

                    self.sW[k]=0.9*self.sW[k]+0.1*self.W_grad[k]
                    self.sb[k]=0.9*self.sb[k]+0.1*self.b_grad[k]
                    self.sGamma[k]=0.9*self.sGamma[k]+0.1*self.gamma_grad[k]
                    self.sBeta[k]=0.9*self.sBeta[k]+0.1*self.beta_grad[k]

                    self.rW[k]=0.999*self.rW[k]+0.001*self.W_grad[k]**2
                    self.rb[k]=0.999*self.rb[k]+0.001*self.b_grad[k]**2
                    self.rGamma[k]=0.999*self.rGamma[k]+0.001*self.gamma_grad[k]**2
                    self.rBeta[k]=0.999*self.rBeta[k]+0.001*self.beta_grad[k]**2

                    self.W[k]=self.W[k]-10*self.learning_rate*self.sW[k]/np.sqrt(1000*self.rW[k]+0.00001)
                    self.b[k] = self.b[k] -10* self.learning_rate * self.sb[k] / np.sqrt(1000*self.rb[k] + 0.00001)
                    self.Gamma[k]=self.Gamma[k]-10*self.learning_rate*self.sGamma[k]/np.sqrt(1000*self.rGamma[k]+0.00001)
                    self.Beta[k]=self.Beta[k]-10*self.learning_rate*self.sBeta[k]/np.sqrt(1000*self.rBeta[k]+0.00001)



    def nn_predict(self,batch_x):
        # 预测
        batch_x=batch_x.T
        m=np.size(batch_x,1)
        self.a[0]=batch_x

        for k in range(self.depth-1):

            y=self.W[k].dot(self.a[k])+np.repeat(self.b[k],m,1)
            if  self.batch_normalization:
                y = (y - self.E[k]) / (self.S[k] + 0.0001)
                y=self.Gamma[k]*y+self.Beta[k]

            target_output_function = {'sigmoid': self.Sigmoid, 'tanh': np.tanh,
                                      'relu': self.Relu, 'softmax': self.SoftMax}
            target_active_function = {'sigmoid': self.Sigmoid, 'tanh': np.tanh,
                                      'relu': self.Relu}
            if k == self.depth - 2:

                self.a[k+1]=target_output_function[self.output_function](y)
            else:

                self.a[k + 1] = target_active_function[self.active_fun](y)

        return None

    def nn_test(self,test_x,test_y):

        self.nn_predict(test_x)
        y_output=self.a[-1].T
        label=np.argmax(y_output,1)
        expect=np.argmax(test_y,1)
        index=label==expect
        success_ratio=np.sum(index)/np.size(index)
        return success_ratio

    def set_input(self,train_x,train_y,validation_x,validation_y,total_iteration=2000,loop_iterations=10,batch_size=100):

        self.train_x=train_x
        self.train_y=train_y
        self.validation_x=validation_x
        self.validation_y=validation_y
        self.total_iteration=total_iteration
        self.loop_iterations=loop_iterations
        self.batch_size=batch_size


    def fit(self):
        # 开始拟合
       max_accuracy=0
       total_accuracy = []
       Total_cost=[]
       accuracy=0

       for item in range(self.total_iteration):
           self.nn_train(self.train_x, self.train_y, self.loop_iterations, self.batch_size)
           Cost=np.array(self.cost)
           Total_cost.append(np.sum(Cost)/np.size(Cost))
           accuracy=self.nn_test(self.validation_x,self.validation_y)
           total_accuracy.append(accuracy)

           if accuracy>max_accuracy:
               max_accuracy=accuracy
               np.save('W.npy', self.W)
               np.save('b.npy', self.b)
               if self.batch_normalization:
                   np.save('Gamma.npy', self.Gamma)
                   np.save('Beta.npy', self.Beta)
           cost_c=Total_cost[item]
           print('Accuracy is %f \n' % accuracy)
           print('Current cost is %f'% cost_c)

       return accuracy







