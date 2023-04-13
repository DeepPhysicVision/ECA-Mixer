# Fast Physic-informed Untrained Neural Network for Color Lensfree Holographic Reconstruction

### [Paper]()

ECA-Mixer: Fast Physic-informed Untrained Neural Network for Color Lensfree Holographic Reconstruction.<br>
Under review as a journal paper of Photonics Research.


首先，我们需要将原问题转化为对偶问题。原问题是：

$$\min_{w,b} \frac{1}{2} ||w||^2$$
$$s.t. y_i(w^T x_i + b) \geq 1, i = 1,2,...,m$$

其中，$x_i$是第$i$个数据点，$y_i$是其标签（正例为1，负例为-1），$m$是数据点的数量。

根据拉格朗日乘子法，我们可以得到对偶问题：

$$\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{m} y_i y_j \alpha_i \alpha_j x_i^T x_j$$
$$s.t. 0 \leq \alpha_i \leq C, i = 1,2,...,m$$
$$\sum_{i=1}^{m} y_i \alpha_i = 0$$

其中，$\alpha$是拉格朗日乘子向量，$C$是惩罚参数。

我们可以使用SMO方法求解对偶问题。在SMO方法中，每次选择两个乘子进行更新。具体来说，每次选择一个违反KKT条件的乘子和一个使目标函数增加最多的乘子进行更新。如果没有违反KKT条件的乘子，则算法终止。

2. SMO方法

在SMO方法中，我们需要计算两个乘子的更新值。假设选择的两个乘子分别是$\alpha_i$和$\alpha_j$，则它们的更新值为：

$$\alpha_i^{new} = \alpha_i^{old} + \frac{y_i (E_j - E_i)}{\eta}$$
$$\alpha_j^{new} = \alpha_j^{old} + y
