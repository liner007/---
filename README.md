# 贝叶斯分类的两个实验
  机器学习导引课程论文-贝叶斯分类。涉及两个实验，一个针对mnist手写数据，一个针对鸢尾花。
  采用python3.6与相关的工具库环境。
  
（1）Python环境需求

①numpy 1.18.5

②TensorFlow 2.3.1

③scikit-learn 0.23.1

④matplotlib 3.2.2

（2）数据集

采用的数据集，都直接在代码中实现了下载
①MNIST手写数据集
这里采用的数据是MINIST手写数据集。MNIST数据集是机器学习领域中非常经典的一个数据集，
由60000个训练样本和10000个测试样本组成，每个样本都是一张28 * 28像素的灰度手写数字图片。
手写体数字识别作为模式识别的一个重要分支,它主要研究如何通过计算机智能地识别出不同场景下的阿拉伯数字。
目前,手写体数字识别技术被广泛应用于互联网、金融、教育等行业。

整个实验是在TensorFlow环境下进行的，同时TensorFlow库提供了MNIST手写数据集。
在具体的实验代码中，可以进行修改的参数主要包括训练集与测试集的规模，选取的类别数目以及特征个数。

②鸢尾花数据集
Iris 鸢尾花数据集是一个经典数据集，在统计学习和机器学习领域都经常被用作示例。
数据集内包含 3 类花，记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
可以通过这4个特征预测鸢尾花卉属于哪一品种。

（3）实验结果

①MNIST手写数据集分类实验

实验号	F1-score	Recall	Precision	Accuracy

一	    0.18	    0.25	   0.14	    0.825

二	    0.33	    0.27	   0.44	    0.87

三	    0.48	    0.48	   0.49	    0.91


②鸢尾花数据集分类试验

F1-score	Recall	Precision	Accuracy

0.45	    0.64	    0.34	   0.74
