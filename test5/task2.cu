#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
using namespace std;

void random_ints(float* a, int n) 
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 5.0);

    for (int i = 0; i < n; i++) 
    {
        a[i] = dis(gen);
    }
}

int main()
{
    int M, N, K;
    cin >> M >> N >> K;
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    int size_a = M * N * sizeof(float);
    int size_b = N * K * sizeof(float);
    int size_c = M * K * sizeof(float);

    A = (float *)malloc(size_a);
    random_ints(A, M * N);
    B = (float *)malloc(size_b);
    random_ints(B, N * K);
    C = (float *)malloc(size_c);
    
    // 分配设备内存
    cudaMalloc((void**)&d_A, size_a);
    cudaMalloc((void**)&d_B, size_b);
    cudaMalloc((void**)&d_C, size_c);
    
    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_A, A, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_b, cudaMemcpyHostToDevice);

    // 创建cublas句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // 记录开始时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, 
                &alpha, d_A, M, d_B, N, &beta, d_C, M);

    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 将数据从设备内存复制到主机内存
    cudaMemcpy(C, d_C, size_c, cudaMemcpyDeviceToHost);
    cout << "Part Result: " << C[2233]  << "  " << C[6044] << "  " << C[M * K - 1] << endl;
    cout << "Used time: " << time << "ms" << endl;

    cublasDestroy(handle);
    free(A);
    free(B);
    free(C);

    return 0;
}