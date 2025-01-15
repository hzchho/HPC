#include<iostream>
#include<mpi.h>
#include<cstring>
#include<random>
#include<ctime>
#include<chrono>
using namespace std;
//自定义结构体
struct Matrix {
    int rows;
    int cols;
    double data[8][8];
};
Matrix mat;
//建立MPI类型
void Build_mpi_type(MPI_Datatype &mpi_matrix_type) {
    MPI_Aint array_of_displacements[3]={0};
    MPI_Datatype array_of_types[3]={MPI_INT, MPI_INT, MPI_DOUBLE};
    MPI_Aint addr1, addr2, addr3;
    //计算每个元素的地址
    MPI_Get_address(&mat.rows,&addr1);
    MPI_Get_address(&mat.cols,&addr2);
    MPI_Get_address(&mat.data,&addr3);

    // array_of_displacements[0]=offsetof(Matrix, rows);
    // array_of_displacements[1]=offsetof(Matrix, cols);
    // array_of_displacements[2]=offsetof(Matrix, data);
    //计算偏移量
    array_of_displacements[1]=addr2-addr1;
    array_of_displacements[2]=addr3-addr2;

    int array_of_blocklengths[3]={1, 1, 8*8};
    //构造自定义MPI结构体
    MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacements, array_of_types, &mpi_matrix_type);
    MPI_Type_commit(&mpi_matrix_type);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int my_rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    Matrix A, B, C;
    //初始化
    if (my_rank==0){
        A.rows=8;
        A.cols=8;
        B.rows=8;
        B.cols=8;
        
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){
                A.data[i][j]=5.28/(i+1);
                B.data[i][j]=6.33/(j+1);
            }
        }
    }

    //创建自定义 MPI数据类型，将结构体元素整理成按连续的地址空间排列
    MPI_Datatype mpi_matrix_type;
    Build_mpi_type(mpi_matrix_type);
    
    auto start=MPI_Wtime();
    //广播A和B结构体
    MPI_Bcast(&A, 1, mpi_matrix_type, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B, 1, mpi_matrix_type, 0, MPI_COMM_WORLD);
    //将矩阵分块
    int part_row=A.rows/comm_sz;
    int start_row=my_rank*part_row;
    int end_row=start_row+part_row;
    
    //矩阵运算
    //每个进程进行特定行的矩阵乘法即可
    for(int i=start_row;i<end_row;i++){
        for(int j=0;j<B.cols;j++){
            C.data[i][j]=0.0;
            for (int k=0;k<A.cols;k++){
                C.data[i][j]+=A.data[i][k]*B.data[k][j];
            }
        }
    }

    //结果回传
    if(my_rank!=0){
        MPI_Send(&C.data[start_row][0], (end_row-start_row)*B.cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }else{
        for(int p=1;p<comm_sz;p++){
            int recv_start_row=p*part_row;
            int recv_end_row=recv_start_row+part_row;
            MPI_Recv(&C.data[recv_start_row][0], (recv_end_row-recv_start_row)*B.cols, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    auto end=MPI_Wtime();
    auto my_time=end-start;
    
    if(my_rank==0){
        cout << "Matrix Multiplication Finished" << endl;
        printf("Used time: %.4fms\n",1000*my_time);
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++){
                cout << C.data[i][j] << " ";
            }
            cout << endl;
        }
    }

    //清理自定义类型
    MPI_Type_free(&mpi_matrix_type);
    
    MPI_Finalize();
    return 0;
}
