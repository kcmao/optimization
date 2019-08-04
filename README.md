# optimization
一些关于算法优化，并行计算的记录

# 评价计算指标和性能指标

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

 

# 带宽   
带宽应用的领域非常多，可以用来标识信号传输的数据传输能力、标识单位时间内通过链路的数据量、标识显示器的显示能力。  
**对于存储设备而言，带宽指的是传输能力，指单位时间内能够传输的数据量大小,更直白一点就是数据传输的速度。**    
   
总线带宽指的是总线在单位时间内可以传输的数据总量，等于总线位宽与工作频率的乘积。例如：对于64位、800MHz的前端总线，它的数据传输率就等于     64bit×800×1000×1000Hz÷8(Byte)÷1024÷1024÷1024≈6.0GB/s   
    
内存带宽指的是内存总线所能提供的数据传输能力。例如：DDR400内存的数据传输频率为400MHz，那么单条模组就拥有       64bit×400×1000×1000Hz÷8(Byte)÷1024÷1024÷1024≈3.0GB/s的带宽。   

**n内存带宽 = 数据传输速率 x 数据总线宽度** 

## 常见内存设备的带宽范围：   
   > 磁盘：  

   >内存：   
   >>RAM:    
   >>>SRAM: 
   >>>>SRAM: cache: 1-10ns

   >>>DRAM:(动态ram)      
   >>>>SDRAM:(synchronous DRAM)同步dram     
   >>>>>DDR:(double data rate sdram)双倍速率SDRAM   
   >>>>>>DDR4：20GB/S    
   
      
   >显存：  
   >内存与显存：  
   
   
# 常见芯片的算力  
以前算力都用flops表示，这是表示浮点运算次数，现在用得是ops:  
GPU:    10 TFLOPS    

ARMV8:

INTEL CPU:   1 TFLPOPS

## cycle
??  



## TX2上yolo优化方法  

**看的网上的**
速度优化的方向：

1、减少输入图片的尺寸， 但是相应的准确率可能会有所下降  
2、优化darknet工程源代码（去掉一些不必要的运算量或者优化运算过程）  
3、剪枝和量化yolov3网络（压缩模型---> 减枝可以参考tiny-yolo的过程 ， 量化可能想到的就是定点化可能也需要牺牲精度）  
4、darknet -----> caffe/tensorflow + tensorrt（主要是针对GPU这块的计算优化）

cat 、proc/cpuinfo
```
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
```

# 关于计算机程序的知识点   
**对于编程语言里的变量/参数/指针/引用等各种难以搞懂的问题，如果理解计算机体系结构，汇编语言，就能比教深刻和正确的理解**。
## 函数空间(程序的构造)   
1. 代码段/数据段/堆/栈/BSS
2. 一个程序在内存中的构造，各部分在内存中的位置，哪些是编译时分配的，哪些是运行时分配的。
3. 变量的定义/声明不同，它在程序中的位置就不同。   
3.1 常量/全局变量/全局常量/局部变量/定义未初始化变量/定义并初始化变量  
3.2    
## 栈帧（fp）
1. 调用子函数会保存当前栈的位置，这个位置就是栈帧
## sp(fp)/pc(lr)寄存器   
1. 成对理解，用来保存一个程序的各种位置，程序栈（sp）,栈帧（fp）,程序当前位置（pc）,程序返回位置（lr）。
## 参数传递（实参/形参）
0. 硬件结构上程序调用的传递方式（编写非汇编程序不用关心）：寄存器传递和压栈传递   
1. 编程语言上三种传递：值传递/指针传递/引用传递。  
1.1 分别代表了三种不同的传递模式，对于理解程序语言很有帮助。
    ```
    值传递：对实参进行拷贝，给形参
    指针传递：对指针变量进行拷贝，给形参。而指针指向的地址是不变的
    引用传递：直接将实参的地址传递，省略掉了指针变量这个值。
    ```
    ```
    引用的规则：   
    （1）引用被创建的同时必须被初始化（指针则可以在任何时候被初始化）。   
    （2）不能有NULL引用，引用必须与合法的存储单元关联（指针则可以是NULL）。   
    （3）一旦引用被初始化，就不能改变引用的关系（指针则可以随时改变所指的对象）。 

    指针传递的实质：

    指针传递参数本质上是值传递的方式，它所传递的是一个地址值。值传递过程中，被调函数的形式参数作为被调函数的局部变量处理，

    即在栈中开辟了内存空间以存放由主调函数放进来的实参的值，从而成为了实参的一个副本。值传递的特点是被调函数对形式参数的

    任何操作都是作为局部变量进行，不会影响主调函数的实参变量的值。（这里是在说实参指针本身的地址值不会变）如果理解不了大可跳过这段

    指针传递和引用传递一般适用于：

    函数内部修改参数并且希望改动影响调用者。对比指针/引用传递可以将改变由形参“传给”实参（实际上就是直接在实参的内存上修改，

    不像值传递将实参的值拷贝到另外的内存地址中才修改）。

    另外一种用法是：当一个函数实际需要返回多个值，而只能显式返回一个值时，可以将另外需要返回的变量以指针/引用传递

    给函数，这样在函数内部修改并且返回后，调用者可以拿到被修改过后的变量，也相当于一个隐式的返回值传递吧.
    ```

2. 引申：指针和引用的区别。   
    指针：也是一个用一块区域存储的值，或者也称其为变量，指针变量：只不过这块区域保存的是另一块区域的地址。   
    引用：仅仅是一个别名。   
    **注意** ：任何程序在编译之后那些变量名称，常量名称都没有了，到了汇编级别仅仅是对某块地址的存取到寄存器和寄存器内加减的操作。    
    理解这个很重要，可加深对程序的理解。   

## 程序在内存中的保存   
1. 代码/数据在程序里的连续保存，如果要读取出来，必须指定位数，比如：   
    int: 8 ，一个byte   
    fp32：32, 4个byte   
    如果位数读错了，读的就是错的，这些都是由程序员指定   

## gdb调试   

## cpu 核
这个涉及到cpu的封装工艺/制造工艺，不同的方式产生不同的效果，适用于不同的场景，  
一个cpu一般包含它的核（逻辑运算单元/控制单元/l1cache（现代cpu封装到cpu内）/总线）   
深入的涉及到（cpu板） （cpu die）等。

### 超线程(SMT:simulate multithreading)（hyper-threading）
*我的电脑 8核16线程，这就是使用了超线程技术*   
超线程技术不同于多线程技术，多线程技术是从软件层面实现的，而超线程技术是通过硬件层面，通俗的说就是给一个核多分了点其他的东西:逻辑控制单元，l1 cache，这样模拟出了两个核的感觉，就是在一个die上多了一点东西  


### 多核（同构）


### 多路（SMP）
