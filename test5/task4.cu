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
// 随机初始化
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
        for (int i = d; i < n - d; i++) 
        {
            for (int j = d; j < n - d; j++)
            {
                a[c * n * n + i * n + j] = 1.1 * dis(gen);
            }
        }
    }
}
// 卷积运算
__global__ void conv_im2col(float *mat, float *res, int m)
{
    int s_block = 9 * m * m;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m)
    {
        float sum = 0.0;
        for(int i = 0; i < 9; i++)
        {
            sum += filter[i] * mat[row * (9 * m) + i * m + col];
            sum += filter[i] * mat[s_block + row * (9 * m) + i * m + col];
            sum += filter[i] * mat[2 * s_block + row * (9 * m) + i * m + col];
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
    int stride = 1;
    n = n + 2 * padding;
    int m = (n - 3) / stride + 1;

    // 生成随机浮点数
    float *mat = (float *)malloc(3 * n * n * sizeof(float));
    random_ints(mat, n, padding);
    float *res = (float *)malloc(m * m * sizeof(float));
    float *im_mat = (float *)malloc(9 * m * m * 3 * sizeof(float));
    
    float *d_mat, *d_res;
    // 分配内存
    cudaMalloc((void **)&d_mat, 9 * m * m * 3 * sizeof(float));
    cudaMalloc((void **)&d_res, m * m * sizeof(float));

    dim3 Block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    int m_block = n * n;
    int s_block = 9 * m * m;
    // 记录开始时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 应该包括将原图像矩阵转化成im2col所用的矩阵的时间
    // 将每个3x3的方块变成一个1x9的列向量，按照步长逐个分解后变成新的矩阵
    for (int c = 0; c < 3; c++)
    {
        int r = 0, l = 0;
        for (int i = 0; i < n - 2; i += stride)
        {
            for (int j = 0; j < n - 2; j += stride)
            {
                im_mat[c * s_block + r * m + l] = mat[c * m_block + i * n + j];
                im_mat[c * s_block + (r + 1) * m + l] = mat[c * m_block + i * n + j + 1];
                im_mat[c * s_block + (r + 2) * m + l] = mat[c * m_block + i * n + j + 2];
                im_mat[c * s_block + (r + 3) * m + l] = mat[c * m_block + (i + 1) * n + j];
                im_mat[c * s_block + (r + 4) * m + l] = mat[c * m_block + (i + 1) * n + j + 1];
                im_mat[c * s_block + (r + 5) * m + l] = mat[c * m_block + (i + 1) * n + j + 2];
                im_mat[c * s_block + (r + 6) * m + l] = mat[c * m_block + (i + 2) * n + j];
                im_mat[c * s_block + (r + 7) * m + l] = mat[c * m_block + (i + 2) * n + j + 1];
                im_mat[c * s_block + (r + 8) * m + l] = mat[c * m_block + (i + 2) * n + j + 2];
                l += 1;
            }
            r += 9;
            l = 0;
        }
    }

    // 将数据从主机复制到设备
    cudaMemcpy(d_mat, im_mat, 9 * m * m * 3 * sizeof(float), cudaMemcpyHostToDevice);
    conv_im2col<<<Grid, Block>>>(d_mat, d_res, m);

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

    cout << "First 5x5 changed_mat:" << endl;
    for (int i = 0; i < 9; i++)
    {
        for (int c = 0; c < 3; c++)
        {
            for (int j = 0; j < 5; j++)
            {
                cout << setw(8) << im_mat[c * s_block + i * m + j] << " ";
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
    free(im_mat);
    free(res);
    cudaFree(d_mat);
    cudaFree(d_res);

    return 0;
}