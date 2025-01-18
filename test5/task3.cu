#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <iomanip>
using namespace std;

#define BLOCK_SIZE 16

// 卷积核
__constant__ float filter[9]=
{
    1.5, -2.5, 1.5,
    -2.5, 4.0, -2.5,
    1.5, -2.5, 1.5
};
// 初始化图像矩阵，并加上填充
void random_ints(float* a, int n, int d) 
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 255);
    
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < n; j++)
            {
                a[c * n * n + i * n + j] = 0.0;
            }
        }
    }

    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < n - d; i++) 
        {
            for (int j = 0; j < n - d; j++)
            {
                a[c * n * n + i * n + j] = 1.1 * dis(gen);
            }
        }
    }
}
// 卷积函数
__global__ void conv2d(float *mat, float *res, int n, int m, int padding, int stride)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m)
    {
        float sum = 0.0;
        for (int kr = 0; kr <= 2; kr++)
        {
            for (int kc = 0; kc <= 2; kc++)
            {
                int i = row * stride + kr;
                int j = col * stride + kc;
                if (i >= 0 && i < n && j >= 0 && j < n)
                {
                    sum += mat[i * n + j] * filter[kr * 3 + kc];
                    sum += mat[n * n + i * n + j] * filter[kr * 3 + kc];
                    sum += mat[2 * n * n + i * n + j] * filter[kr * 3 + kc];
                }
            }
        }
        res[row * m + col] = sum;
    }
}

int main()
{
    // 图像规模
    int n;
    cin >> n;
    // 填充
    int padding = 1;
    // 步长
    int stride = 3;
    n = n + 2 * padding;
    int m = (n - 3) / stride + 1;
    
    // 生成随机浮点数
    float *mat = (float *)malloc(3 * n * n * sizeof(float));
    random_ints(mat, n, padding);
    float *res = (float *)malloc(m * m * sizeof(float));
    
    float *d_mat, *d_res;
    // 分配内存
    cudaMalloc((void **)&d_mat, 3 * n * n * sizeof(float));
    cudaMalloc((void **)&d_res, m * m * sizeof(float));
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_mat, mat, 3 * n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 记录开始时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // 卷积
    conv2d<<<Grid, Block>>>(d_mat, d_res, n, m, padding, stride);

    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 将数据从设备复制到主机
    cudaMemcpy(res, d_res, m * m * sizeof(float), cudaMemcpyDeviceToHost);
    
    cout << "Padding: " << padding << " Stride: " << stride << endl;
    cout << "Input:" << endl;
    for (int i = padding; i < min(n - padding, 5 + padding); i++)
    {
        for (int c = 0; c < 3; c++)
        {
            for (int j = padding; j < min(n - padding, 5 + padding); j++)
            {
                cout << setw(8) << mat[c * n * n + i * n + j] << " ";
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
            cout << setw(8) << res[i * m + j] << " ";
        }
        cout << endl;
    }
    cout << "Used time: " << time << "ms" << endl;

    free(mat);
    free(res);
    cudaFree(d_mat);
    cudaFree(d_res);

    return 0;
}