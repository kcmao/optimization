# openblas
BLAS(Basic Linear Algebra Subprograms)是一组线性代数计算中通用的基本运算操作函数集合。   
openblas是基于其发展的一个开源实现。作者张先轶。   
相关知识整理如下：  
[github网址](https://github.com/xianyi/OpenBLAS)  

[openblas官网](http://www.openblas.net)  

### 有用的相关网址  
作者的一次汇报：[雷锋网](https://www.leiphone.com/news/201704/Puevv3ZWxn0heoEv.html)  

openblasapi整理[openblas api](https://blog.csdn.net/weixin_43800762/article/details/87811697)  

openblas 整体框架解读[csdn](https://blog.csdn.net/zzk1995/article/details/70991878)  
* github项目 
  github [how to optimize gemm](https://github.com/flame/how-to-optimize-gemm)  
* 知乎 gemm 算法解读   
  [openblas gemm入门](https://zhuanlan.zhihu.com/p/65436463)    
  [gemm cache优化](https://zhuanlan.zhihu.com/p/69700540)    
  [通用矩阵乘（GEMM）优化与卷积计算](https://zhuanlan.zhihu.com/p/66958390)
* openblas与其他各个框架的性能对比   
[在回答中](https://www.zhihu.com/question/27872849)

##带宽

#define vmlaq_laneq_f32(a, b, v, lane)  vmlaq_n_f32(a, b, vgetq_lane_f32(v, lane)) 
