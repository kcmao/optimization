# CUDA学习记录  
很多地方借鉴参考和学习了Ewenwan的github,非常感谢。

* CUDA（Compute Unified Device Architecture）的中文全称为计算统一设备架构。

[CUDA编程之快速入门!!!!!推荐](https://www.cnblogs.com/skyfsm/p/9673960.html)

[参考1](https://www.cnblogs.com/cuancuancuanhao/category/1236273.html)

[参考2](https://blog.csdn.net/fishseeker/article/details/75093166)

[参考3](https://bbs.csdn.net/topics/390798229)

[The CMake version of cuda_by_example ](https://github.com/Ewenwan/cuda_by_example)

[CUDA Program：CUDA image rgb to gray；CUDA KLT](https://github.com/canteen-man/CUDA-Program)

[CUDA 编程 加速 计算机视觉 ！！！推荐](https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA)

[并行编程 CUDA openMP 等](http://heather.cs.ucdavis.edu/~matloff/158/PLN/ParProcBook.pdf)

[并行编程 CUDA openMP 等 中文版](https://github.com/thirdwing/ParaBook)

# 1.GPU架构特点
    高性能计算的关键是利用多核处理器进行并行计算。
    当我们求解一个计算机程序任务时，我们很自然的想法就是将该任务分解成一系列小任务，把这些小任务一一完成。在串行计算时，我们的想法就是让我们的处理器每次处理一个计算任务，处理完一个计算任务后再计算下一个任务，直到所有小任务都完成了，那么这个大的程序任务也就完成了。
![](https://img2018.cnblogs.com/blog/1093303/201809/1093303-20180919122904566-1040268509.png)
    
    串行计算的缺点非常明显，如果我们拥有多核处理器，我们可以利用多核处理器同时处理多个任务时，而且这些小任务并没有关联关系（不需要相互依赖，比如我的计算任务不需要用到你的计算结果），那我们为什么还要使用串行编程呢？为了进一步加快大任务的计算速度，我们可以把一些独立的模块分配到不同的处理器上进行同时计算（这就是并行），最后再将这些结果进行整合，完成一次任务计算。下图就是将一个大的计算任务分解为小任务，然后将独立的小任务分配到不同处理器进行并行计算，最后再通过串行程序把结果汇总完成这次的总的计算任务。

![](https://img2018.cnblogs.com/blog/1093303/201809/1093303-20180919122917935-1661200386.png)

**所以，一个程序可不可以进行并行计算，关键就在于我们要分析出该程序可以拆分出哪几个执行模块，这些执行模块哪些是独立的，哪些又是强依赖强耦合的，独立的模块我们可以试着设计并行计算，充分利用多核处理器的优势进一步加速我们的计算任务，强耦合模块我们就使用串行编程，利用串行+并行的编程思路完成一次高性能计算。**

    GPU和CPU的不同硬件特点决定了他们的应用场景，CPU是计算机的运算和控制的核心，GPU主要用作图形图像处理。图像在计算机呈现的形式就是矩阵，我们对图像的处理其实就是操作各种矩阵进行计算，而很多矩阵的运算其实可以做并行化，这使得图像处理可以做得很快，因此GPU在图形图像领域也有了大展拳脚的机会。下图表示的就是一个多GPU计算机硬件系统，可以看出，一个GPU内存就有很多个SP和各类内存，这些硬件都是GPU进行高效并行计算的基础。

![](https://img2018.cnblogs.com/blog/1093303/201809/1093303-20180919122932879-1946399786.png)

    现在再从数据处理的角度来对比CPU和GPU的特点。CPU需要很强的通用性来处理各种不同的数据类型，比如整型、浮点数等，同时它又必须擅长处理逻辑判断所导致的大量分支跳转和中断处理，所以CPU其实就是一个能力很强的伙计，他能把很多事处理得妥妥当当，当然啦我们需要给他很多资源供他使用（各种硬件），这也导致了CPU不可能有太多核心（核心总数不超过16）。而GPU面对的则是类型高度统一的、相互无依赖的大规模数据和不需要被打断的纯净的计算环境，GPU有非常多核心（费米架构就有512核），虽然其核心的能力远没有CPU的核心强，但是胜在多，
    在处理简单计算任务时呈现出“人多力量大”的优势，这就是并行计算的魅力。

**整理一下两者特点就是**：
* 1.CPU：擅长流程控制和逻辑处理，不规则数据结构，不可预测存储结构，单线程程序，分支密集型算法
* 2.GPU：擅长数据并行计算，规则数据结构，可预测存储模式

![](https://img2018.cnblogs.com/blog/1093303/201809/1093303-20180919122947035-1099878851.png)

# CUDA存储器类型：

	每个线程拥有自己的 register寄存器 and loacal memory 局部内存
	每个线程块拥有一块 shared memory 共享内存
	所有线程都可以访问 global memory 全局内存
	
	还有，可以被所有线程访问的
	     只读存储器：
	     constant memory (常量内容) and texture memory
        
	a. 寄存器Register
	   寄存器是GPU上的高速缓存器，其基本单元是寄存器文件，每个寄存器文件大小为32bit.
	   Kernel中的局部(简单类型)变量第一选择是被分配到Register中。
           特点：每个线程私有，速度快。
	
	b. 局部存储器 local memory
	
	   当register耗尽时，数据将被存储到local memory。
	   如果每个线程中使用了过多的寄存器，或声明了大型结构体或数组，
	   或编译器无法确定数组大小，线程的私有数据就会被分配到local memory中。
	   
           特点：每个线程私有；没有缓存，慢。
           注：在声明局部变量时，尽量使变量可以分配到register。如：
           unsigned int mt[3];
           改为：　unsigned int mt0, mt1, mt2;
	
	c. 共享存储器 shared memory
           可以被同一block中的所有线程读写
           特点：block中的线程共有；访问共享存储器几乎与register一样快.
	
	d. 全局存储器 global memory
　         特点：所有线程都可以访问；没有缓存
	
	e. 常数存储器constant memory
	   用于存储访问频繁的只读参数
	   特点：只读；有缓存；空间小(64KB)
	   注：定义常数存储器时，需要将其定义在所有函数之外，作用于整个文件
	
	f. 纹理存储器 texture memory
           是一种只读存储器，其中的数据以一维、二维或者三维数组的形式存储在显存中。
	   在通用计算中，其适合实现图像处理和查找，对大量数据的随机访问和非对齐访问也有良好的加速效果。
           特点：具有纹理缓存，只读。

# threadIdx，blockIdx, blockDim, gridDim之间的区别与联系

     在启动kernel的时候，要通过指定gridsize和blocksize才行
     dim3 gridsize(2,2);   // 2行*2列*1页 形状的线程格，也就是说 4个线程块
        gridDim.x，gridDim.y，gridDim.z相当于这个dim3的x，y，z方向的维度，这里是2*2*1。
	序号从0到3，且是从上到下的顺序，就是说是下面的情况:
	 具体到 线程格 中每一个 线程块的 id索引为：
	 
        grid 中的 blockidx 序号标注情况为：      0     2 
                                               1     3
					       
       dim3 blocksize(4,4);  // 线程块的形状，4行*4列*1页，一个线程块内部共有 16个线程
     
       blockDim.x，blockDim.y，blockDim.z相当于这个dim3的x，y，z方向的维度，
       这里是4*4*1.序号是0-15，也是从上到下的标注：

	block中的 threadidx 序号标注情况   0       4       8      12 
					  1       5       9      13
					  2       6       10     14
					  3       7       11     15
	1.  1维格子，1维线程块，N个线程======

	 实际的线程id tid =  blockidx.x * blockDim.x + threadidx.x
	 块id   0 1 2 3 
	 线程id 0 1 2 3 4
	2. 1维格子，2D维线程块
	 块id   1 2 3 
	 线程id  0  2
	        1  3
			       块id           块线程总数
	 实际的线程id tid =  blockidx.x * blockDim.x * blockDim.y + 
			       当前线程行数    每行线程数
			    threadidx.y * blockDim.x   + 
			       当前线程列数
			    threadidx.x


