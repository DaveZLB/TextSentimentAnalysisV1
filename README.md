# 文本情感分类（Sentiment Classifier）

- 基于机器学习（ML）的方法

 Bayes, SVM ,DecisionTree, KNN
 
- 基于深度学习（Deep Learning）的方法

 MLP, CNN, RNN(LSTM)

# 使用说明

代码基于Python3.6+,下面的pip指的是pip3，python 指的是python3

- 安装依赖
```cmd
pip install -r requirement.txt
```

- 生成WordEmbedding

```cmd
python process_data.py
```
- 生成词向量

```cmd
python gen_word_to_vec.py
```

- 分别执行对应算法python文件


# 其他说明
1、语料  
电影评论，训练集合20000（正向10000，负向10000）  
电影评论，测试集合20000（正向3000，负向3000）  
2、语料处理  
使用jieba进行分词  
3、输入向量化  
使用预先训练的wordvector.bin文件进行向量化  
对于传统机器学习算法，要求输入的是N维向量， 采用句子向量求和平均  
对于CNN，RNN深度学习算法，要求输入的是N*M维向量，分别对应查找并生成向量  

# 训练精度

| Algorithm | Accuracy |
| --- | --- |
| DecisionTree | 0.6907302434144715 |
| Bayes | 0.7437479159719906 |
| KNN | (n=14)0.7909303101033678 |
| SVM | 0.8302767589196399 |
| MLP | (20epoches) 0.8359 |
| CNN | (20epoches) 0.8376 |
| LSTM | (20epoches) 0.8505 |
