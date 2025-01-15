#include<iostream>
#include<mpi.h>
#include<cstring>
#include<random>
#include<ctime>
#include<chrono>
using namespace std;

void matrix_multiply(int m,int n,int k,double *A,double *B,double *C){
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            C[i*k+j]=0.0;
            for(int l=0;l<n;l++){
                C[i*k+j]+=A[i*n+l]*B[l*k+j];
            }
        }
    }
}

void matrix_initialize(double *matrix,int m,int n){
    random_device rd;
    mt19937 generator(rd());
    
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            uniform_real_distribution<> distribution(0.0,5.0);
            matrix[i*n+j]=0.05;
        }
    }
}

int main(int argc, char *argv[]){
    int comm_sz;
    int my_rank;
    string str;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    int M=512*2*2;
    int N=512*2*2;
    int K=512*2*2;
 
    double max_time;
    double min_time;
    double avg_time;
    double *A=NULL;
    double *B=new double[N*K];
    double *C=new double[M*K];
    double *local_A=new double[(M/comm_sz)*N];
    double *local_C=new double[(M/comm_sz)*K];

    if(my_rank==0){
        A=new double[M*N];
        matrix_initialize(A,M,N);
        matrix_initialize(B,N,K); 
    }

    auto start=MPI_Wtime();
    MPI_Scatter(A, (M/comm_sz)*N, MPI_DOUBLE, local_A, (M/comm_sz)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // auto start1=chrono::high_resolution_clock::now();
    MPI_Bcast(B, N*K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // auto start2=chrono::high_resolution_clock::now();
    matrix_multiply(M/comm_sz, N, K, local_A, B, local_C);
    // auto start3=chrono::high_resolution_clock::now();
    MPI_Gather(local_C, (M/comm_sz)*K, MPI_DOUBLE, C, (M/comm_sz)*K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto end=MPI_Wtime();
    auto my_time=end-start;

    MPI_Reduce(&my_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    printf("Process %d: Used time: %.4fms\n",my_rank,1000*my_time);
    
    if(my_rank==0){
        cout << "Matrix Multiplication Finished" << endl;
        printf("Used time:\nMax_time: %.4fms\nMin_time: %.4fms\nAvg_time: %.4fms\n",1000*max_time,1000*min_time,1000*avg_time/comm_sz);
        delete []A;
    }

    delete []local_A;
    delete []local_C;
    delete []B;
    delete []C;

    MPI_Finalize();
    return 0;
}