#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <random>
#include <iomanip> 
using namespace std;

void random_ints(float* a, int n) 
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0.0, 255.0);

    for (int i = 0; i < n; i++) 
    {
        a[i] = 1.1 * dis(gen);
    }
}


int main()
{
    int n;
    cin >> n;
    // 初始化cudnn
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // 定义数据维度
    int batch_size = 1;
    int in_channels = 3;
    int in_height = n;
    int in_width = n;
    int out_channels = 1;
    int fileter_height = 3;
    int filter_width = 3;
    int stride = 3;
    int padding = 1;
    int m = (n + 2 * padding - fileter_height) / stride + 1;
    int out_height = m;
    int out_width = m;

    // 定义输入输出描述符
    cudnnTensorDescriptor_t input_Desc, output_Desc;
    cudnnFilterDescriptor_t filter_Desc;
    cudnnConvolutionDescriptor_t conv_Desc;

    // 创建描述符
    cudnnCreateTensorDescriptor(&input_Desc);
    cudnnCreateTensorDescriptor(&output_Desc);
    cudnnCreateFilterDescriptor(&filter_Desc);
    cudnnCreateConvolutionDescriptor(&conv_Desc);

    // 设置描述符
    cudnnSetTensor4dDescriptor(input_Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
    batch_size, in_channels, in_height, in_width);    
    
    cudnnSetTensor4dDescriptor(output_Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    batch_size, out_channels, out_height, out_width);

    cudnnSetFilter4dDescriptor(filter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    out_channels, in_channels, fileter_height, filter_width);
    
    cudnnSetConvolution2dDescriptor(conv_Desc, padding, padding, stride, stride, 1, 1, 
    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    // 分配设备内存、
    float *d_input, *d_output, *d_filter;
    size_t input_size = batch_size * in_channels * in_height * in_width * sizeof(float);
    size_t output_size = batch_size * out_channels * out_height * out_width * sizeof(float);
    size_t filter_size = out_channels * in_channels * fileter_height * filter_width * sizeof(float);
    float *input = (float*)malloc(input_size);
    float *output = (float*)malloc(output_size);
    random_ints(input, 3 * n * n);
    float filter[27]=
    {
        1.5, -2.5, 1.5,
        -2.5, 4.0, -2.5,
        1.5, -2.5, 1.5,
        1.5, -2.5, 1.5,
        -2.5, 4.0, -2.5,
        1.5, -2.5, 1.5,
        1.5, -2.5, 1.5,
        -2.5, 4.0, -2.5,
        1.5, -2.5, 1.5
    };

    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_output, output_size);
    cudaMalloc((void**)&d_filter, filter_size);

    // 将数据拷贝到设备
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filter_size, cudaMemcpyHostToDevice);

    // 创建cudnn激活函数描述符
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // 记录开始时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 前向传播
    cudnnConvolutionForward(handle, &alpha, input_Desc, d_input, filter_Desc, d_filter, conv_Desc,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, NULL, 0, &beta, output_Desc, d_output);

    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 将结果拷贝到主机
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    
    cout << "Padding: " << padding << " Stride: " << stride << endl;
    cout << "Input:" << endl;
    for (int i = 0; i < min(n, 5); i++)
    {
        for (int c = 0; c < 3; c++)
        {
            for (int j = 0; j < min(n, 5); j++)
            {
                cout << setw(8) << input[c * n * n + i * n + j] << " ";
            }
            if (c != 2)
            {
                cout << "|";
            }
        }
        cout << endl;
    }
    
    cout << "Result:" << endl;
    for (int i = 0; i < min(5, m); i++)
    {
        for (int j = 0; j < min(5, m); j++)
        {
            cout << setw(8) << output[i * m + j] << " ";
        }
        cout << endl;
    }
    cout << "Used time: " << time << "ms" << endl;

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    free(input);
    free(output);

    // 销毁描述符
    cudnnDestroyTensorDescriptor(input_Desc);
    cudnnDestroyTensorDescriptor(output_Desc);
    cudnnDestroyFilterDescriptor(filter_Desc);
    cudnnDestroyConvolutionDescriptor(conv_Desc);
    cudnnDestroy(handle);

    return 0;
}
