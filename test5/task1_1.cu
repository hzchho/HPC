#include <iostream>
#include <cuda_runtime.h>
#include <random>
using namespace std;

#define BLOCK_SIZE 32

void random_ints(float* a, int n) 
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 5.0);

    for (int i = 0; i < n; i++) 
    {
        a[i] = dis(gen);
        // a[i] = 1.0 * i;
    }
}
// 优化后的矩阵乘法
__global__ void mat_mul(float *A, float *B, float *C, int M, int N, int K) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 用于存储当前线程块所需要的矩阵块
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) 
    {
        if (row < M && t * BLOCK_SIZE + threadIdx.x < N) 
        {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * BLOCK_SIZE + threadIdx.x];
        } 
        else 
        {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < K && t * BLOCK_SIZE + threadIdx.y < N) 
        {
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * K + col];
        } 
        else 
        {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        // 用于同步
        __syncthreads();

        float sum = 0;
        for (int i = 0; i < BLOCK_SIZE; i++) 
        {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();

        if (row < M && col < K) 
        {
            C[row * K + col] += sum;
        }
    }
}

int main() 
{
    // 查看设备属性
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cout << "Number of thread blocks: " << prop.maxGridSize[0] << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    
    int M, N, K;
    cin >> M >> N >> K;
    float *A, *B, *C; 
    float *d_A, *d_B, *d_C;

    int size_a = M * N * sizeof(float);
    int size_b = N * K * sizeof(float);
    int size_c = M * K * sizeof(float);

    // 分配设备内存
    cudaMalloc((void**)&d_A, size_a);
    cudaMalloc((void**)&d_B, size_b);
    cudaMalloc((void**)&d_C, size_c);

    // 分配主机内存以及随机初始化
    A = (float*)malloc(size_a);
    random_ints(A, M * N);
    B = (float*)malloc(size_b);
    random_ints(B, N * K);
    C = (float*)malloc(size_c);

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_A, A, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_b, cudaMemcpyHostToDevice);
    
    // 设置线程块和网格大小
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 记录开始时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 进行矩阵乘法
    mat_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 将数据传回主机
    cudaMemcpy(C, d_C, size_c, cudaMemcpyDeviceToHost);
    cout << "Part Result: " << C[2233]  << "  " << C[6044] << "  " << C[M * K - 1] << endl;
    cout << "Used time: " << time << "ms" << endl;

    // 释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}