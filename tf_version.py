#机器学习项目步骤
#1 界定问题

#导入必须的库

#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def get_data(): 
    train = pd.read_csv("input/train.csv",index_col = "PassengerId")
    test = pd.read_csv("input/test.csv", index_col="PassengerId")
    return train,test

#年龄处理
def age_preproecssing(dataframe):
    imputer = Imputer(strategy='mean', axis=0)
    imputer.fit(train[['Age']]) #总是用训练数据进行训练
    dataframe.Age = imputer.transform(dataframe[['Age']])
    return dataframe

#船票处理
def fare_preproecssing(dataframe):
    imputer = Imputer(strategy='median', axis=0)
    imputer.fit(train[['Fare']])
    dataframe.Fare = imputer.transform(dataframe[['Fare']])
    return dataframe

# 1 性别编码
def sex_encoding(dataframe):
    sex_encoder = LabelEncoder()
    sex_encoder.fit(train['Sex'])
    dataframe['Sex_labeled'] = sex_encoder.transform(dataframe['Sex'])
    return dataframe

#Embarked的另一种方法
def add_embarkd_cols(dataframe):
    dataframe['C']=0
    dataframe['Q']=0
    dataframe['S']=0
    dataframe.loc[dataframe["Embarked"]=='C', "C"] = 1
    dataframe.loc[dataframe["Embarked"]=='Q', "Q"] = 1
    dataframe.loc[dataframe["Embarked"]=='S', "S"] = 1
    return dataframe

#从名字中提取身份信息 并转换成onehot
def add_title_cols(dataframe):
    dataframe['Title'] = [name.split(", ")[1].split(". ")[0] for name in dataframe.Name]
    #Conditional that returns a boolean Series with column labels specified
    dataframe.loc[~dataframe.Title.isin(['Mr','Miss','Mrs']),'Title']='Others'
    dataframe['is_Mr']=0
    dataframe['is_Miss']=0
    dataframe['is_Mrs']=0
    dataframe['is_Others']=0
    dataframe.loc[dataframe["Title"]=='Mr', "is_Mr"] = 1
    dataframe.loc[dataframe["Title"]=='Miss', "is_Miss"] = 1
    dataframe.loc[dataframe["Title"]=='Mrs', "is_Mrs"] = 1
    dataframe.loc[dataframe["Title"]=='Others', "is_Others"] = 1
    return dataframe

#家庭人数
def add_fimaly_size_col(dataframe):
    dataframe["FamilySize"] = dataframe["SibSp"] + dataframe["Parch"] + 1
    return dataframe

#预处理最后一步 特征缩放
def feature_scaler(data):
    std_scaler = MinMaxScaler()
    std_scaler.fit(X_train)
    data_scaled = std_scaler.fit_transform(data)
    return data_scaled

train,test = get_data()
train = age_preproecssing(train)
test = age_preproecssing(test)
#删除房间号字段
train = train.drop('Cabin',axis=1)
test = test.drop('Cabin',axis=1)
#填充登船点
train.Embarked=train.Embarked.fillna('C')
test.Embarked=test.Embarked.fillna('C')

train = fare_preproecssing(train)
test = fare_preproecssing(test)

train = sex_encoding(train)
test = sex_encoding(test)

train = add_embarkd_cols(train)
test = add_embarkd_cols(test)


train = add_title_cols(train)
test = add_title_cols(test)
    
train = add_fimaly_size_col(train)
test = add_fimaly_size_col(test)

usable_coulumns = [ 'Pclass', 'Age','Fare', 'Sex_labeled', 'C', 'Q', 'S', 'is_Mr','is_Miss', 'is_Mrs', 'is_Others', 'FamilySize']
X_train = train[usable_coulumns]
y_train = train['Survived'].values.reshape(-1,1)
X_test = test[usable_coulumns]


X_train_scaled = feature_scaler(X_train)
X_test_scaled = feature_scaler(X_test) 


tf.reset_default_graph()

n_inputs = X_train_scaled.shape[1]
n_layer1 = 50
n_outputs = 1
keep_prob = .5
learning_rate = .03

he_init = tf.variance_scaling_initializer()

with tf.name_scope('input') as scope:
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")#[981*12]
    y = tf.placeholder(tf.float32, shape=(None), name="y")#[981 * 1]

with tf.name_scope("dnn") as scope:
    layer1 = tf.layers.dense(X, n_layer1, activation=tf.nn.relu, name="layer1",kernel_initializer=he_init)#[12*300]
    drop1 = tf.nn.dropout(layer1,keep_prob,name='dropout1')
    layer2 = tf.layers.dense(layer1, n_layer2, activation=tf.nn.relu, name="layer2",kernel_initializer=he_init)#[300*100]
    drop2 = tf.nn.dropout(layer1,keep_prob,name='dropout2')
    layer3 = tf.layers.dense(layer2, n_layer3, activation=tf.nn.relu, name="layer3",kernel_initializer=he_init)#[100*10]
    drop3 = tf.nn.dropout(layer1,keep_prob,name='dropout3')
    logits = tf.layers.dense(layer3, n_outputs, name="logits")#[10*1]
    outputs = tf.round(tf.nn.sigmoid(logits),name='outputs')

with tf.name_scope("loss"):#sigmoid_cross_entropy_with_logits
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

batch_size = 100

#获取每一批数据


loss_dic = []
with tf.Session() as sess:
    init.run()
    for epoch in range(10000):
        for batch_index in range(X_train_scaled.shape[0]//batch_size):
            X_train_batch = get_batch_data(X_train_scaled,batch_index,epoch)
            y_train_batch = get_batch_data(y_train,batch_index,epoch)
            sess.run(training_op, feed_dict={X: X_train_batch, y: y_train_batch})
        #没100次迭代输出loss（总体）
        if epoch % 100 == 0:
            _loss = sess.run(loss, feed_dict={X: X_train_scaled, y: y_train})
            loss_dic.append(_loss)
            print('EPOCH:',epoch," LOSS:",_loss)
    saver.save(sess,'tf_model/titanic_model.ckpt')
    
with tf.Session() as sess:
    saver.restore(sess,'tf_model/titanic_model.ckpt')
    y_predict = sess.run(outputs,feed_dict={X: X_test_scaled})
    
with tf.Session() as sess:
    init.run()
    y_predict = sess.run(outputs,feed_dict={X: X_train_scaled})
    
    
def output_csv(predict_data):
    submission = pd.read_csv("input/gender_submission.csv", index_col="PassengerId")
    submission.Survived = predict_data
    submission.to_csv('output/my_final_submission_with_dropout_minmax_scaled.csv', index=True)

y_predict = best_svm_clf_reload.predict(X_test_scaled)
output_csv(y_predict)




def plot_loss(loss_dic): 
    plt.plot(np.arange(len(loss_dic)),loss_dic)
    #plt.ylim([0.5,1])
    plt.show()


def get_batch_data(data,batch_index,epoch_index):
    np.random.seed(epoch_index)
    random_index = np.random.permutation(len(data))
    inds = random_index[batch_index * batch_size:batch_index * batch_size + batch_size]
    data_batch = data[inds]
    return data_batch

#验证模型
from sklearn.metrics import accuracy_score
accuracy_score(y_true,y_pred)




    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    