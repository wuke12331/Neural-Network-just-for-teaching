from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from NN import nn_Creat
import pandas

# filename='training.csv'
# names=['1', '2', '3', '4', '5', '6','class']
# data=read_csv(filename,delimiter=' ',names=names)
# x,y=np.split(data,(6,),1)

def convter(s):
    dic1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,'draw': 0}
    dic1.setdefault(s,1)
    return dic1[s]

filenames='krkopt.csv'
names=['wang1_x', 'wang1_y', 'wang2_x', 'wang2_y', 'bing_x', 'bing_y', 'class']
# names=['1', '2', '3', '4', '5', '6', 'class']

dataframe=read_csv(filenames,delimiter=',',names=names,converters={0:convter,2:convter,4:convter,6:convter})
# dataframe=read_csv(filenames,delimiter=' ',names=names)
x,y=np.split(dataframe,(6,),1)

scaler=StandardScaler().fit(x)
x=scaler.transform(x)
test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(x, np.ravel(y), test_size=test_size,
random_state=seed)
Y_train=pandas.get_dummies(Y_train).values
Y_test=pandas.get_dummies(Y_test).values

NN=nn_Creat([6,10,10,10,10,10,2],active_fun='tanh',learning_rate=0.01,batch_normalization=1,output_function='MSE',
            optimization_method='normal',weight_decay=0)
NN.set_input(X_train,Y_train,X_test,Y_test,batch_size=100)
NN.fit()




