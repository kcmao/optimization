<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


# 激活函数/损失函数

1. relu/sigmod/softmax/交叉熵等的区别   

```
一方面：sigmod和rulu都是激活函数，用在神经网络的后面起到非线性的作用。

另一方面：relu的出现是完全为了从激活函数的角度改进而提出的。
sigmod不经能作为非线性的激活作用，同时由于其是将输出压缩到0-1之间，更有概率的味道。和softmax一样成为一种输出数据到概率的映射作用。 

所以sigmod不仅可以作为卷积的激活部分，同时可以作为最后输出到概率的映射，和softmax一样。   

交叉熵作为一种计算损失的一个公式。用来衡量预测的输出和真值的区别。
```

继续relu,relu的稀疏作用？对量化的影响
```
relu能够将输出小于0的都变成0，使得输出很稀疏，然后下一层就很稀疏。越来越稀疏，不知道作用是什么，优缺点是什么？  

relu的存在使得量化的方式有所不同，
一般量化分为权值量化和激活值量化，如果激活函数是relu，那么直接量化到0-255，加入激活值是sigmod或其他呢，使得输出可以小于0，
```


2. 损失函数    
    1. frcnn损失函数  
交叉熵损失  
smooth l1损失：   
$smooth_{L1}\left ( x \right )= \left\{\begin{matrix}& 0.5x^{2} &if \left | x \right | < 1\\ & \left | x \right |- 0.5 &otherwise\end{matrix}\right.$ 

![](./pic/smooth_l1.png)   
    softmax-cross_entropy损失：  
    $ $

3. yolo损失：  
sigmoid-cross_entropy损失：





