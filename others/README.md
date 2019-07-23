## 关于评价计算指标和性能指标

评价芯片性能用flops, 衡量程序/模型运算量也用这个flops，所以有点晕了，看了好些资料才有些明白  

一个知乎问题问了这个问题,   
[CNN 模型所需的计算力（flops）和参数（parameters）数量是怎么计算的？](https://www.zhihu.com/question/65305385)   
其中一个答案： 
[chen liu](https://www.zhihu.com/question/65305385/answer/451060549)
首先避免不了这个词：flops   

FLOPS：注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。   

FLOPs：注意s小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。   



作者也是如上解释FLOPS概念，是一个衡量硬件性能的指标

在[gemm 从零入门](https://zhuanlan.zhihu.com/p/65436463)这篇知乎中，作者说   
gflops是优化效果的指标。矩阵乘的计算量是2 * M * N * K，拿计算量除以耗时即为当前gemm版本的gflops。如何评估当前还有多大优化空间呢？如果我们能测出芯片的极限gflops，拿芯片的极限和代码的效果做比较，二者接近就说明没有什么提升空间了。如何测试芯片的极限，后续会提供样例

在《并行算法设计与性能优化》这本书中，P48页作者解释了FLOPS: 两个N * N的矩阵相乘，耗时为t.她的FLOPS为2xN3/t,  
作者同时提到一点，用flops衡量程序性能并不适合，因为程序如果降低了算法复杂度的话，其flops反而会降低。按道理是指标越高越好，所以FLOPS比较适用于场合是无法通过算法优化降低运算复杂度，反而可以通过访存等其他优化手段降低运算时间的程序。比如一些运算数量固定不变的程序，比如矩阵乘法等科学计算。因为芯片硬件的浮点运算次数是固定的，当你程序测出来的flops越接近芯片的理论flops,则说明你的程序对硬件的利用率越高。

**注：为什么要计算程序的flops并和芯片理论flops比，这是因为任何程序都很难将cpu的性能完全发挥出来，因为有大量的访存时间导致cpu在等待，所以优化程序的方法诞生出了两个分支：

**大O复杂度：  

一个是宏观上的，也是最有效最直接的，从程序的时间复杂度入手，目的是减少程序的计算量，不管程序指令具体是什么，乘、加、除、判断都认为是一次操作，这样从宏观上评价一个程序的计算量。    

**FLOPS 

另一个是从微观角度出发，当一个程序计算量已经无法再优化了，转而从访存、循环展开、向量化等手段优化时间，尽量让程序更好的利用cpu和存储。使用的是flops指标。

## TX2上yolo优化方法  

**看的网上的**
速度优化的方向：

1、减少输入图片的尺寸， 但是相应的准确率可能会有所下降  
2、优化darknet工程源代码（去掉一些不必要的运算量或者优化运算过程）  
3、剪枝和量化yolov3网络（压缩模型---> 减枝可以参考tiny-yolo的过程 ， 量化可能想到的就是定点化可能也需要牺牲精度）  
4、darknet -----> caffe/tensorflow + tensorrt（主要是针对GPU这块的计算优化）  


[root@dvrdvs /] # cat /proc/cpuinfo
processor       : 0
model name      : ARMv7 Processor rev 1 (v7l)
BogoMIPS        : 2786.91
Features        : swp half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt 
CPU implementer : 0x41
CPU architecture: 7
CPU variant     : 0x1
CPU part        : 0xc0e
CPU revision    : 1

processor       : 1
model name      : ARMv7 Processor rev 1 (v7l)
BogoMIPS        : 2793.47
Features        : swp half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt 
CPU implementer : 0x41
CPU architecture: 7
CPU variant     : 0x1
CPU part        : 0xc0e
CPU revision    : 1

processor       : 2
model name      : ARMv7 Processor rev 1 (v7l)
BogoMIPS        : 2793.47
Features        : swp half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt 
CPU implementer : 0x41
CPU architecture: 7
CPU variant     : 0x1
CPU part        : 0xc0e
CPU revision    : 1

processor       : 3
model name      : ARMv7 Processor rev 1 (v7l)
BogoMIPS        : 2793.47
Features        : swp half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt 
CPU implementer : 0x41
CPU architecture: 7
CPU variant     : 0x1
CPU part        : 0xc0e
CPU revision    : 1

Hardware        : hi3536
Revision        : 0000
Serial          : 0000000000000000

processor　：系统中逻辑处理核的编号。对于单核处理器，则课认为是其CPU编号，对于多核处理器则可以是物理核、或者使用超线程技术虚拟的逻辑核
vendor_id　：CPU制造商      
cpu family　：CPU产品系列代号
model　　　：CPU属于其系列中的哪一代的代号
model name：CPU属于的名字及其编号、标称主频
stepping　  ：CPU属于制作更新版本
cpu MHz　  ：CPU的实际使用主频
cache size   ：CPU二级缓存大小
physical id   ：单个CPU的标号
siblings       ：单个CPU逻辑物理核数
core id        ：当前物理核在其所处CPU中的编号，这个编号不一定连续
cpu cores    ：该逻辑核所处CPU的物理核数
apicid          ：用来区分不同逻辑核的编号，系统中每个逻辑核的此编号必然不同，此编号不一定连续
fpu             ：是否具有浮点运算单元（Floating Point Unit）
fpu_exception  ：是否支持浮点计算异常
cpuid level   ：执行cpuid指令前，eax寄存器中的值，根据不同的值cpuid指令会返回不同的内容
wp             ：表明当前CPU是否在内核态支持对用户空间的写保护（Write Protection）
flags          ：当前CPU支持的功能
bogomips   ：在系统内核启动时粗略测算的CPU速度（Million Instructions Per Second）
clflush size  ：每次刷新缓存的大小单位
cache_alignment ：缓存地址对齐单位
address sizes     ：可访问地址空间位数
