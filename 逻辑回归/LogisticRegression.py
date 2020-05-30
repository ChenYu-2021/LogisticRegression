import numpy as np
import pandas as pd
#pd.set_option('display.max_columns', 100)

data = pd.read_csv('KaggleCredit2.csv')

print(data.shape)
# 查看数据是否有缺失
print(data.info())
print(data.isnull().sum())

# 去掉为空的样本数据(也就是去掉为空的行)
data.dropna(axis=0, inplace=True)
print(data.shape)

# 得到X和y
X = data.drop(['SeriousDlqin2yrs'],axis=1) # 按列删除
y = data['SeriousDlqin2yrs']


# 1.将数据分成训练集和测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(x_train.shape, x_test.shape)

# 2. 使用逻辑回归进行数据预测
from sklearn.linear_model import LogisticRegression
'''
LogisticRegression参数说明
multi_class: one-vs-rest
solver: 优化损失函数的算法，sag为随机平均梯度下降
class_weight: 用于标示分类模型中各种类型的权重， balanced为不输入

'''
lr_model = LogisticRegression(multi_class='ovr', solver='sag', class_weight='balanced')
lr_model.fit(x_train, y_train)
score = lr_model.score(x_train, y_train) # 得到预测准确率(用分类正确的个数除以总的训练样本数)
print(score)

# 3. 在测试集上进行预测,计算准确度
# 准确率：逻辑回归将类别分成两类：正类和负类，准确率就是在正类的样本中被识别为正类的概率，而回归率是在正类和负类中被识别为正类的概率
from sklearn.metrics import accuracy_score
train_score = accuracy_score(y_train, lr_model.predict(x_train))
test_score = lr_model.score(x_test, y_test)
print('训练集准确度：', train_score)
print('测试集准确度：', test_score)

# 4. 召回率：在所有正类别样本中，被正确识别为正类别的比例是多少
from sklearn.metrics import recall_score
train_recall = recall_score(y_train, lr_model.predict(x_train), average='macro')
test_recall = recall_score(y_test, lr_model.predict(x_test), average='macro')
print('训练集的回归率：', train_recall)
print('测试集的回归率：', test_recall)

'''
在logistic regression当中，一般我们的概率判定边界为0.5，但是我们可以把阈值设定低一些，来提高模型的“敏感度”，试试看把阈值设定为0.3，再看看这时的评估指标(主要是准确率和召回率)
'''
y_pro = lr_model.predict_proba(x_test) # 获取预测概率
y_prd2 = [list(p > 0.3).index(1) for i, p in enumerate(y_pro)]
train_score = accuracy_score(y_test, y_prd2)
print(train_score)