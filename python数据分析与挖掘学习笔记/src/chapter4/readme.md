* [4.1 数据清洗](#41-数据清洗)
    * [4.1.1 缺失值处理](#411-缺失值处理)

## chapter4 数据预处理
1. 预处理 = 数据清洗 + 数据集成 + 数据转换 + 数据规约
2. 在数据挖掘的过程中，数据预处理工作量占到了整个过程的60%

### 4.1 数据清洗
1. 任务：删除原数据集中的无关数据、重复数据，平滑噪声数据，筛选与挖掘主题无关的数据，处理缺失值、异常值

#### 4.1.1 缺失值处理
1. 处理方法：**删除记录、数据插补、不处理**
2. 常见数据插补方法：

| 插补方法 | 方法描述 |
| :---: | :---: |
| 均值/中位数/众数 | 根据属性值的类型，用该属性取值的平均值/中位数/众数进行插补 |
| 固定值 | 将缺失的值用一个常量替换 |
| 最近邻 | 在记录中找到与缺失样本最接近的样本的该属性值插补 |
| 回归方法 | 对带有缺失值的变量，根据已有的数据和与其有关的其它变量（因变量）的数据建立拟合模型来预测缺失的属性值 |
| 插值法 | 利用已知点建立合适的差值函数 $f(x)$，未知值由对应点 $x$ 求出的函数值 $f(x)$ 近似代替

3. 插值法
    - 拉格朗日插值和牛顿插值，本质上其结果是一样的，只是表示形式不同。而后者相对于前者而言，具有承袭性和易于变动节点的特点
    - 拉格朗日插值法：已知平面上 $n$ 个点可以找到一个 $n-1$ 次多项式 $y=a_0+a_1x+a_2x^2+\cdots+a_{n-1}x^{n-1}$，使得多项式可以过这 $n$ 个点。
        - 推导过程：
            1. 设函数：$y=a_0+a_1x+a_2x^2+\cdots+a_{n-1}x^{n-1}$
            2. 将已知的 $n$ 个点代入该方程得：
            $$
                \begin{cases}
                y_1=a_0+a_1x_1+a_2x_1^2+\cdots+a_{n-1}x_1^{n-1} \\
                y_2=a_0+a_1x_2+a_2x_2^2+\cdots+a_{n-1}x_2^{n-1} \\
                \cdots \\
                y_n=a_0+a_1x_n+a_2x_n^2+\cdots+a_{n-1}x_n^{n-1}
                \end{cases}
            $$
            解出拉格朗日插值多项式为
            $$
                \begin{array}{l}
                L(x)=y_1\frac{(x-x_2)(x-x_3)\cdots(x-x_n)}{(x_1-x_2)(x_1-x_3)\cdots(x_1-x_n)} \\
                \\+ y_2\frac{(x-x_1)(x-x_3)\cdots(x-x_n)}{(x_2-x_1)(x_2-x_3)\cdots(x_2-x_n)} \\
                \\+ \cdots \\
                \\+ y_n\frac{(x-x_1)(x-x_2)\cdots(x-x_{n-1})}{(x_n-x_1)(x_n-x_2)\cdots(x_n-x_{n-1})} \\
                = \sum_{i=0}^ny_i\prod_{j=0,j \neq i}^n\frac{x-x_j}{x_i-x_j}
                \end{array}
            $$
        - 使用方法：将缺失的函数值对应的点 $x$ 
        - 代入到拉格朗日插值多项式就可以得到带缺失值的近似值 $L(x)$，然后用该值来近似替代对应的未知值
        - 缺点：在实际中，当插值节点增减时，拉格朗日插值多项式就会变化，需要重新求解，很不方便。牛顿插值法则用于克服这一问题
        - 所属包：`from scipy.interpolate import lagrange`，其中`lagrange(y, x)`返回值是一个拉格朗日插值函数，使用时直接传参即可。
    - 牛顿插值法：
        - 推导过程：
            1. 求已知的 $n$ 个点对 $(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)$ 的所有阶差商公式
            $$
                \begin{matrix}
                f[x_1,x]=\frac{f[x]-f[x_1]}{x-x_1}=\frac{f(x)-f(x_1)}{x-x_1}
                \end{matrix}
            $$
            $$
                \begin{matrix}
                f[x_2,x_1,x]=\frac{f[x_1,x]-f[x_2,x_1]}{x-x_2}
                \end{matrix}
            $$
            $$
                \begin{matrix}
                f[x_3,x_2,x_1,x]=\frac{f[x_2,x_1,x]-f[x_3,x_2,x_1]}{x-x_3}
                \end{matrix}
            $$
            $$\cdots$$
            $$
                \begin{matrix}
                f[x_n,x_{n-1},\cdots,x_1,x]=\frac{f[x_{n-1},\cdots,x_1,x]-f[x_n,\cdots,x_1]}{x-x_n}
                \end{matrix}
            $$
            2. 联立以上差商公式建立如下插值多项式 $f(x)$
            $$
                \begin{eqnarray*}
                f(x) & = &f(x_1)+ \\
                & & (x-x_1)f[x_2,x_1]+ \\
                & & (x-x_1)(x-x_2)f[x_3,x_2,x_1]+ \\
                & & (x-x_1)(x-x_2)(x-x_3)f[x_4,x_3,x_2,x_1]+ \\
                & & \cdots+ \\
                & & (x-x_1)(x-x_2)\cdots(x-x_{n-1})f[x_n,\cdots,x_1] + \\
                & & (x-x_1)(x-x_2)\cdots(x-x_n)f[x_n,\cdots,x_1,x] \\
                & = & P(x)+R(x)
                \end{eqnarray*}
            $$
            其中
            $$
                \begin{array}{l}
                P(x) & = &f(x_1)+ \\
                & & (x-x_1)f[x_2,x_1]+ \\
                & & (x-x_1)(x-x_2)f[x_3,x_2,x_1]+ \\
                & & (x-x_1)(x-x_2)(x-x_3)f[x_4,x_3,x_2,x_1]+ \\
                & & \cdots+ \\
                & & (x-x_1)(x-x_2)\cdots(x-x_{n-1})f[x_n,\cdots,x_1] \\
                \\
                R(x) & = &(x-x_1)(x-x_2)\cdots(x-x_n)f[x_n,\cdots,x_1,x]
                \end{array}
            $$
            $P(x)$ 是牛顿插值逼近函数，$R(x)$ 是误差函数
        - 使用方法：将缺失的函数值对应的点带入插值多项式得到缺失值的近似值 $f(x)$
    - [相关代码](code/lagrange_interp.py)
    - **在插值前先对进行异常值检测，然后将之置为空，然后在后面插值时进行值预测**

#### 4.1.2 异常值处理
1. 异常值常见处理方法，**方法选取视情况而定**：

| 异常值处理方法 | 方法描述 |
| :---: | :---: |
| 删除含有异常值的记录 | |
| 置为缺失值 | 置为缺失值，利用缺失值处理的方法进行处理 |
| 品均值修正 | 利用前后两个观测值的均值来修正该值 |
| 不处理 | 直接在具有异常值的数据集上进行挖掘建模 |

###  4.2 数据集成
1. 定义：将多个数据源合并存放在一个**一致**的数据存储中的过程
2. 需解决的问题：**实体识别**和**属性冗余**

#### 4.2.1 实体识别
1. 定义：指从不同数据源中识别出现实世界的实体，任务是**统一不同源数据的矛盾**。<br />
常见矛盾如下：
    - 同名异义：如数据源A的属性ID和数据源B的属性ID分别描述菜品编号和订单编号，即描述的是不同的实体
    - 异名同义：如数据源A的sales_dt和数据源B的sales_data都是描述销售日期的，即 A.sales_dt=B.sales_data，二者为同一实体
    - 单位不统一

#### 4.2.2 冗余属性识别
- 同一属性多次出现
- 同一属性命名不一致导致重复

### 4.3 数据变换
1. 定义：将数据规范化，转成“适当”形式，以满足挖掘任务及算法实现的需要

#### 4.3.1 简单函数变换
1. 指对原始数据进行某些数学函数变换，常见的包括平方、开方、取对数、差分运算（$ \nabla f(x)=f(x_{k+1})-f(x_k) $）等
2. 常用来将不具有正态分布的数据/非平稳序列变换成具有正态分布的数据/平稳序列

#### 4.3.2 规范化（归一化）
- 目的：数据的不同评价指标往往具有不同的量纲，数值间的差别可能很大，不进行处理可能会影响到数据分析的结果。故**为了消除指标之间的量纲和取值方位差异的影响**，需要进行标准化处理，将数据按照比例进行缩放，使之落入一个特定的区域，便于进行综合分析
- 最大-最小规范化
    - 离差标准化，是一种对原始数据的线性变换，将数值映射到$[0,1]$之间
    - 公式：$x^*=\frac{x-min}{max-min}$
    - 缺点：
        1. 若$max$很大，则规范化后的值会接近于0，且相差不大
        2. 若将来遇到超过$[min,max]$取值范围的时候，则会引起系统出错
- 零-均值规范化
    - 标准差标准化，经过处理的数据的均值为0，标准差为1
    - 公式：$x^*=\frac{x-\arg{x}}{\sigma}$
    - 当前使用最多的数据标准化方法
- 小数定标规范化
    - 通过移动属性值的小数位数，将属性值映射到$[-1,1]$之间，移动的小数位数取决于属性值绝对值的最大值
    - 公式：$x^*=\frac{x}{10^k}$

#### 4.3.3 连续属性离散化
1. 目的：一些数据挖掘算法（如某些分类算法——ID3、Apriori）要求数据是分类属性形式，故常常需要将连续属性换成分类属性
2. 离散化过程：在数据的取值范围内设定若干个离散的划分点，将取值范围划分为一些离散化的区间，最后用不同的符号或整数值代表落在每个子区间中的数据之
3. 常用的离散化方法：
    - 等宽法
        + 将属性的值域3划分为具有相同宽度的区间
        + 缺点：对离群点比较敏感，倾向于不均匀地把属性值分不到每个区间（即有些区间包含很多数据，有些则很少），会严重损坏建立的决策模型
    - 等频法
        + 将记录按升序排序，然后规定每n条记录划分为一组
        + 缺点：可能将相同的数据值划分到不同的区间
    - 基于聚类分析的方法
        + 将连续属性的值用聚类算法（如K-Means）进行聚类，然后将聚类得到的簇进行处理，核定得到一个簇的连续属性值并做同一标记。（需用户指定簇的个数）
    - [样例代码](code/data_discretization)

#### 4.3.4 属性构造
1. 利用已知属性集构造出新的属性并加入到现有的属性集合中
2. 目的：提取更有用的信息，挖掘更深层次的模式，提高挖掘结果的精度
3. [样例代码](code/line_rate_construct.py)

#### 4.3.5 小波变换
1. 提供了一种非平稳信号的时频（时域和频域）分析手段，可以有粗及细地逐步观察信号，从中提取有用的信息。即：**能够刻画某个问题的特征量往往是隐含在一个信号的某个或者某些分量中，小波变换能把非平稳信号分解问表达不同层次、不同频带信息的数据序列，即小波系数。选取适当的小波系数，即完成了信号的特征提取**
2. 参考链接：
    - 博客 [http://blog.jobbole.com/101976/](http://blog.jobbole.com/101976/)
    - [个人对小波变换的粗浅理解](../../../原理记录/小波变换.pdf)