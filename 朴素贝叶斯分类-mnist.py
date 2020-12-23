import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
 
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
train_num = 1000 #训练集个数   这里可以修改
test_num = 200 #这里是测试集个数   种类也可以修改
class_num = 10
desimon = 784


def calculate(label_true,label_pre): #除了计算acc  这里计算了其他的一些参数 F1 召回率等
    RR = 0 #标签为真  模型判真
    FF = 0 #标签为假  模型判假
    RF = 0 #标签为真  模型判假
    FR = 0 #标签为假  模型判真
    for i in range(len(label_true)):
        if label_pre[i] == label_true[i] and label_true[i] == 1:
            RR += 1
        elif label_pre[i] == label_true[i] and label_true[i] == 0:
            FF += 1
        elif label_pre[i] != label_true[i] and label_true[i] == 1:
            RF += 1
        else:
            FR += 1
    # print(RR)
    # print(FF)
    # print(RF)
    # print(FR)
    recall = RR / (RR + RF)
    precision = RR / (RR + FR)
    F1 = 2 * precision * recall / (precision + recall)
    print("F1-Score :",F1)
    print("Precision:",precision)
    print("Recall   :",recall)


x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels
prediction = []
for i in range(test_num):
    test = x_test[i]
    class_rate = []
    # 求每一个类别的概率，这里MNIST数据集共有10个类别
    for j in range(class_num):
        # 找到样本中类别是j的下标
        class_is_j_index = np.where(y_train[:train_num] == j)[0]
        # 类别是j的比率
        j_rate = len(class_is_j_index)/len(y_train)
        # 取出类别是j的样本
        class_is_j_x = np.array([x_train[x] for x in class_is_j_index])
        # 遍历每个维度
        for k in range(desimon):
            # 找到j类样本集中该维度下的值与测试样本中该维度的值的差小于0.8的样本，并求占j类样本的比率，与j_rate依次相乘
            # 这里我规定的界限是0.8，因为MNIST中样本数字在0到1之间，并且是两端分布，要么是0，要么接近1。
            j_rate *= len([item for item in class_is_j_x if np.fabs(item[k] - test[k]) < 0.8])*1.0 / len(class_is_j_x)
        class_rate.append(j_rate)
    # 找到贝叶斯预测值最大的类别，作为该测试的预测类别，放到结果集中
    prediction.append(np.argmax(class_rate))
    print(i, 'prediction:', prediction[-1], 'actual:', y_test[i])
 
calculate(y_test[:test_num],prediction)

accurancy = np.sum(np.equal(prediction, y_test[:test_num])) / test_num
print('accurancy:', accurancy)


