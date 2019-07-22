# Learning notes  
一些乱七八糟的深度学习知识：    
1. 逻辑回归/线性回归/感知机/全连接层的区别与联系？   
    1. 多层感知机是叫mlp  
    2. 逻辑回归和多层感知机？   
    ```
    逻辑回归是一层感知机带上一个激活函数
    ```   
    3. 逻辑回归和线性回归都叫做线性模型   

    ```  
   我觉得是及其相似的产物，是不同时间段的产物，其都是线性模型y=wx+b基础上加上一个激活函数或者不同损失函数的构成物，逻辑回归/线性回归/感知机是单独作为一个算法，用于分类或者回归，全连接是在卷积出来后作为其中一层。
    ``` 
2. 全连接/1x1卷积的计算原理    
    [cnn](./cnn.md) 
```
全连接其实完全就是一个参数矩阵W乘以一个输入向量X的操作gemv。如果输入大于一个batch,就是gemm:W [X1,X2],其中W是矩阵， X1,X2分别是向量。而1x1卷积其实和全连接一样，也是完全相当于gemm。
```

3. **卷积操作的本质**？  
    [cnn](./cnn.md)

    **从不同维度看卷积，卷积到底是个啥玩意儿**？    
    [cnn](./cnn.md)   


4. relu/sigmod/softmax/交叉熵等的区别？   
    [activation](./activation_loss.md)

5. 继续relu,relu的稀疏作用？对量化的影响？  
    [activation](./activation_loss.md)

6. 其他激活函数如何量化？  
    **TODO**


## TODO.
### 机器学习
1. 

### 深度学习
1. BN/GN等等正则化方法。
2. 目标检测:    
    1. roi pooling/ roi align **TODO**
    2. nms  **TODO**
    3. darknet  **TODO** 
    4. 
3. attention机制 **TODO**

4. 上采样的几种：  
    反卷积:    
    upsample()  
    unpooling()  
    空洞卷积()  

5. 轻量级网络及卷积网络计算量公式    
    [轻量级网络](./cnn.md)

### 评价指标相关
1. roc/auc/f1 score/  
   图像：  
   iou/miou/map/miss rate-fppi  






