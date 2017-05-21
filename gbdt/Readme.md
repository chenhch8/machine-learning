### 基于GBDT的机器学习
#### 算法总述
- GBDT是boosting算法的一种——将多个弱学习器，通过加法模型，组合成一个强学习器。对于GBDT而言，其弱分类器就是决策树，采用的代价函数就是参差，其核心思想是，通过不断地增加决策树的数量，来使得参**差逐**渐缩小直至为0（实际上达到0的概率很低）。在预测的时候，则通过累加每棵决策树的预测结果，得到的最终结果即为预测值
- 需要注意的是，对于GBDT而言，它的每一棵决策树，都是回归树，而非分类树，因为只用回归树的累加结果才有意义。同时，每棵决策树的叶子节点都是人为固定的。在我的代码中，为了便于实现，还限定了每棵决策树均为完全二叉树。若最中预测的是一个类别，而非一个连续值，则可以设定一个阈值，通过比较阈值与最终预测值的大小，从而再决定是哪个类别
- 决策树包含分类树和回归树

#### GBDT算法流程
1. 初始化：
    - 矩阵 $samples=\sum_{i=0}^N\sum_{j=0}^Mvalue_{ij}$，其中 $value_{ij}$ 表示第 $i$ 个样本的第 $j$ 个特征所对应的特征值
    - 数组 $lists=\sum_{i=0}^Ngoal_i$，其中 $goal_i$ 表示第 $i$ 个样本所对应的类别或者一个值
    - 参差 $errs=list$
2. 开始训练：
训练过程如下
```python
dtrees = []
# 建第 i 棵决策树
For i: 0 -> n
    # 找到当前最优决策树
    tree = findBestTree(samples, lists)
    # 将当前最优决策树加入树集合中
    dtrees.append(tree)
    # 更新残差
    updateErrs(samples, tree, errs)
```
那么应该要如何找到最优决策树？其方法如下：
```python
def findBestTree(samples, lists):
    # 对每个特征
    for feature in Features:
        '''
        根据当前特征的所有特征值，采用‘最小方差’找出当前特征下的最优回归树
        '''
    '''
    比较各个特征对应的‘最小方差’，即‘局部最小方差’，从中找出全局最小方差，则该全局最小方差对应的回归树树就是我们要找的最优决策树
    '''
```
注意，在找最优决策树时，寻找的标准不仅仅有‘最小方差’，还有其它方法，只不过在回归树中，‘最小方差’是最常用的方法只要<br />
那么要如何更新残差？残差其实就是真实值和预测值之间的差值。更新方法如下：
```python
def updateErrs(samples, tree, errs):
    # 对每个样本
    for samples[index] in samples:
        '''[1] 通过tree来计算当前样本 samples[index] 的预测值，记为 i'''
        '''[2] 更新残差 errs[index] -= i'''
```

### 数学推导
- 残差计算：$$ Var(\{Y_1,\dots,Y_l\})=\sum_{j=1}^l\frac{Y_j}{Y}Var(Y_j)=\sum_{j=1}^l\frac{|Y_j|}{|Y|}\left(\frac{1}{|Y_j|}\sum_{y\in Y_j}y^2-\arg^2y_j\right) $$
$$ =\frac{1}{|Y|}\sum_{y\in Y}y^2-\sum_{j=1}^l\frac{|Y_j|}{Y}\arg^2y_j $$
其中$$ Var(Y_j)=\frac{1}{|Y_j|}\sum_{y\in _j}\left(y-\arg y_j\right)^2=\frac{1}{|Y_j|}\sum_{y\in Y_j}y^2-2\arg y_j\sum_{y\in Y_j}y+\sum_{y\in Y_j}\arg^2{y_j} $$

$$=\frac{1}{|Y_j|}\left(\frac{1}{|Y_j|}\sum_{y\in Y_j}y^2-2\arg y_j|Y_j|\arg y_j+|X|\arg^2y_j\right)=\frac{1}{|Y_j|}\sum_{y\in Y_j}y^2-\arg^2y_j $$

其中 $ \arg y_j=\frac{1}{|Y_j|}\sum_{y\in Y_j}y $

### 数据处理注意
- 在寻找最优决策树时，如果要计算的特征值过多，则可以通过随机采样其中若干个来计算，这样能加快运算速度。我是每次随机抽取1000个特征值进行计算

### 目录说明
```
.
├── data
│   ├── sample_submission.txt 
│   ├── test_data.txt ----------------- 测试样本数据集
│   └── train_data.txt ---------------- 训练样本数据集
├── output
│   ├── gbdt_result.json -------------- gbdt模型
│   └── test_result.txt --------------- 测试结果
├── Readme.md
└── src ------------------------------- 源文件
    ├── dtree.py
    ├── gbdt.py
    ├── globalVar.py
    ├── run.py ------------------------ 入口文件
    └── utils.py
```

### 环境说明
- 环境：ubuntu
- 语言：python3.5以上
- 第三方模块：numpy

### 链接
- [训练测试数据集](https://pan.baidu.com/s/1dFGeSgx)

