//mm == 8
//nn == 1
#include <stdio.h>
#include <time.h>
#include "hka_types.h"
//#include "arm_neon.h"
#pragma warning(disable: 4996)
#include "NEON_2_SSE.h"



//复数乘法      (a + ib) * (c + id) = (ac - bd) + i(ad + bc)

//复数共轭乘法  (a + ib) * (c - id) = (ac + bd) + i(bc - ad)

//复数共轭乘法  (a.re + a.im) * (b.re - b.rm) = (a.re*b.re + a.im*b.im) + i(a.im*b.re - a.re*b.im)


//复数共轭乘法宏定义
#define ARTD_RSP_DOA_FC_CONJUGATE_MPY(b,a,c) \
	c.re = a.re * b.re + a.im *b.im; \
	c.im = a.im * b.re - a.re *b.im;

//复数加法宏定义
#define  ARTD_RSP_DOA_FC_ADD(b,a,c) \
	c.re = a.re + b.re; \
	c.im = a.im + b.im;


/***********************************************************************************************************************
* 功  能: 复数矩阵共轭乘法， in * in',定点模式
* 参  数:
*         in                 - I          复数矩阵 [mm * nn]
*         out                - O          输出     [mm * mm]
*         mm                 - I          ina 的 行数
*         nn                 - I          inb 的 列数
* 返回值:
*         空
***********************************************************************************************************************/
//read data 1，读1个文件的数据
static HKA_VOID Read_data(HKA_SC16 *in, HKA_SC32 *out)
{
	FILE *fp, *fp2;
	int i = 0;
	fp = fopen("D:/mkc/neon/doa_data/doa_data/doa_sc_mat_mult_trans_input2", "rb");
	if (fp == NULL)
	{
		printf("读取文件失败\n");
	}

	int ret;
	ret = fread(in, sizeof(HKA_SC16), 8, fp);
	fclose(fp);


	fp2 = fopen("D:/mkc/neon/doa_data/doa_data/doa_sc_mat_mult_trans_output2", "rb");
	if (fp2 == NULL)
	{
		printf("读取文件失败\n");
	}

	ret = fread(out, sizeof(HKA_SC32), 64, fp);
	fclose(fp2);

}

//read data 2 读10个文件的数据
static HKA_VOID Read_data2(HKA_SC16 *in, HKA_SC32 *out, char *input, char *output)
{
	FILE *fp, *fp2;
	int i = 0;
	fp = fopen(input, "rb");
	if (fp == NULL)
	{
		printf("intput file read failed\n");
	}

	int ret;
	ret = fread(in, sizeof(HKA_SC16), 8, fp);
	fclose(fp);


	fp2 = fopen(output, "rb");
	if (fp2 == NULL)
	{
		printf("output file read failed\n");
	}

	ret = fread(out, sizeof(HKA_SC32), 64, fp);
	fclose(fp2);

}

// origin version 
static HKA_VOID  ARTD_RSP_doa_sc_mat_mult_trans(HKA_SC16 *in, HKA_SC32 *out, HKA_S32 mm, HKA_S32 nn)
{
	HKA_S32 m = 0;
	HKA_S32 n = 0;
	HKA_S32 k = 0;
	HKA_SC32 sum = { 0 };
	HKA_SC32 tmp = { 0 };
	HKA_F32  wght = 1.f / nn;


	//for (m = 0; m < mm; m++)   //mm = 8
	for (m = 0; m < mm; m++)
	{
		//for (n = m; n < mm; n++)   
		for (n = m; n < mm; n++)
		{
			sum.re = 0;
			sum.im = 0;

			for (k = 0; k < nn; k++)    // nn = 2
			{	
				//注意：b在前，a在后
				ARTD_RSP_DOA_FC_CONJUGATE_MPY(in[k * mm + n], in[k * mm + m], tmp);
				ARTD_RSP_DOA_FC_ADD(tmp, sum, sum);
			}

			//此处需要对协方差矩阵求均值,这是DML原理需要的
			out[m * mm + n].re = (HKA_S32)(sum.re * wght);
			out[m * mm + n].im = (HKA_S32)(sum.im * wght);
			out[n * mm + m].re = (HKA_S32)(sum.re * wght);
			out[n * mm + m].im = (HKA_S32)(-1.f * sum.im * wght);
		}
	}
}

//version 2: int32  优化版本
static HKA_VOID  ARTD_RSP_doa_sc_mat_mult_trans3(HKA_SC16 *in, HKA_SC32 *out, HKA_S32 mm, HKA_S32 nn)
{

	/*
	 运算次数：8*2*（4+2）次运算
	 8：一共8轮。
	 2：每轮取4个数并行相乘，要做两次。
	 4+2：每次并行操作包含4次乘法，1次加法，1次减法。
	 
	 复数共轭乘法  (a + ib) * (c - id) = (ac + bd) + i(bc - ad)
	*/

	//pt用来控制8轮中每次取的那个数。
	HKA_S16 *pt = in;
	HKA_SC16 *tmp = in;

	//这个暂时没用到
	HKA_F32  wght = 1.f / nn;

	int16x4x2_t vec;           //一次取4个复数，并实部虚部分解。
	int32x4x2_t reg_r;         //保存4个复数共轭乘的结果。
	int16x4_t a, b, c, d;
	int32x4_t ac, bd, bc, ad;

	int i = 0;
	int j = 0;

	for (i = 0; i < mm; i++)
	{	
		//取一个复数，并实部虚部分别复制4份。
		a = vld1_dup_s16(pt);
		b = vld1_dup_s16(pt + 1);

		for (j = 0; j < 2; j++)
		{	
			//一次取4个数,取2次
			vec = vld2_s16(tmp);
			c = vec.val[0];
			d = vec.val[1];
			
			//int32x4x2 reg_r = (ac+bd) + (bc-ad)i
			//ac + bd
			ac = vmull_s16(a, c);
			bd = vmull_s16(b, d);
			reg_r.val[0] = vaddq_s32(ac, bd);

			//bc - ad
			bc = vmull_s16(b, c);
			ad = vmull_s16(a, d);
			reg_r.val[1] = vsubq_s32(bc, ad);

			//将计算完的四个结果存入out
			vst2q_s32(out, reg_r);

			out = out + 4;
			tmp += 4;
		}
		pt = pt + 2;
		tmp = in;
	}
}

//version 3: TODO


// test 1 10组输入数据，测试未优化版本的正确性
static HKA_VOID test_trans(HKA_SC16 *in, HKA_SC32 *out, HKA_SC32 *out1)
{
	//my neon test
	int i = 0;
	int j = 0;
	int k = 0;

	for (k = 1; k < 11; k++)
	{

		char input[256];
		char output[256];
		sprintf(input, "../../doa_data/doa_data/doa_sc_mat_mult_trans_input%d", k);
		sprintf(output, "../../doa_data/doa_data/doa_sc_mat_mult_trans_output%d", k);

		Read_data2(in, out, input, output);

		//int32向量化
		ARTD_RSP_doa_sc_mat_mult_trans(in, out1, 8, 1);

		for (i = 0; i < 64; i++)
		{
			if (out[i].re != out1[i].re || out[i].im != out1[i].im)
			{
				printf("it's not true\n");
			}
			else
			{
				j += 1;
			}
		}
		printf("%d\n", j);   //如果全部输出的output争取，会输出64
		j = 0;
	}
}


// test 2 10组输入数据，测试优化版本的正确性
static HKA_VOID test_trans3(HKA_SC16 *in, HKA_SC32 *out, HKA_SC32 *out3)
{
	//my neon test
	int i = 0;
	int j = 0;
	int k = 0;

	for (k = 1; k < 11; k++)
	{	

		char input[256];
		char output[256];
		sprintf(input, "../../doa_data/doa_data/doa_sc_mat_mult_trans_input%d", k);
		sprintf(output, "../../doa_data/doa_data/doa_sc_mat_mult_trans_output%d", k);
	
		Read_data2(in, out, input, output);

		//int32向量化
		ARTD_RSP_doa_sc_mat_mult_trans3(in, out3, 8, 1);

		for (i = 0; i < 64; i++)
		{
			if (out[i].re != out3[i].re || out[i].im != out3[i].im)
			{
				printf("it's not true\n");
			}
			else
			{
				j += 1;
			}
		}
		printf("%d\n", j);   //如果全部输出的output争取，会输出64
		j = 0;
	}
}



int main()
{
	int i = 0;
    //HKA_SC16 in[16] = { { 1, 1 }, { 1, 2 }, { 2, 2 }, { 2, 2 }, { 3, 3 }, { 3, 2 }, { 4, 4 }, { 4, 2 }, { 5, 5 }, { 5, 2 }, { 6, 6 }, { 6, 2 }, { 7, 7 }, { 7, 2 }, { 8, 8 }, { 8, 2 } };
	HKA_SC16 in[8];

	HKA_SC32 out[64];

	HKA_SC32 out1[64];
	HKA_SC32 out2[64];
	HKA_SC32 out3[64];
	
	long profiler_t0, profiler_t1;	

	//test1 测试原始版本的速度及正确性
	//如果全部输出的output正确，会输出10个64
	profiler_t0 = clock();
	//for (i = 0; i < 1000; i++)
	test_trans(in, out, out1);
	profiler_t1 = clock();
	printf("complex matmul1 cost time is %d\n", profiler_t1 - profiler_t0);
	
	//test2 测试优化版本的速度及正确性
	//如果全部输出的output正确，会输出10个64
	profiler_t0 = clock();
	//for (i = 0; i < 1000;i++)
	test_trans3(in, out, out3);
	profiler_t1 = clock();
	printf("complex matmul2 cost time is %d\n", profiler_t1 - profiler_t0);
	
	return 0;
}

