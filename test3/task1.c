#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>

int M,N,K;
double** A;
double** B;
double** C;
void matrix_multiply(){
    int my_rank=omp_get_thread_num();
    int thread_count=omp_get_num_threads();
    
    int batch=M/thread_count;
    int begin_row=batch*my_rank;
    int end_row=batch*(my_rank+1);

    for(int i=begin_row;i<end_row;i++){
        for(int j=0;j<K;j++){
            C[i][j]=0.0;
            for(int l=0;l<N;l++){
                C[i][j]+=A[i][l]*B[l][j];
            }
        }
    }
}

int main(int argc, char* argv[]){
    int thread_count=strtol(argv[1], NULL, 10);
    scanf("%d %d %d",&M,&N,&K);

    A=malloc(sizeof(double*)*M);
    B=malloc(sizeof(double*)*N);
    C=malloc(sizeof(double*)*M);

    for(int i=0;i<M;i++){
        A[i]=malloc(sizeof(double)*N);
    }
    for(int i=0;i<N;i++){
        B[i]=malloc(sizeof(double)*K);
    }
    for(int i=0;i<M;i++){
        C[i]=malloc(sizeof(double)*K);
    }

    srand(time(NULL));
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            A[i][j]=rand()*5.53/(double)RAND_MAX;
        }
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<K;j++){
            B[i][j]=rand()*3.35/(double)RAND_MAX;
        }
    }


    double start=omp_get_wtime();
    // #pragma omp parallel num_threads(thread_count)
    // matrix_multiply();
    int i=0,j=0,l=0;
    #pragma omp parallel for num_threads(thread_count) shared(A,B,C,M,N,K) private(i,j,l)
    for(i=0;i<M;i++){
        for(j=0;j<K;j++){
            C[i][j]=0.0;
            for(l=0;l<N;l++){
                C[i][j]+=A[i][l]*B[l][j];
            }
        }
    }
    double end=omp_get_wtime();
    
    printf("Used time: %.2lf ms",(end-start)*1000);
    for(int i=0;i<M;i++){
        free(A[i]);
    }
    for(int i=0;i<N;i++){
        free(B[i]);
    }
    for(int i=0;i<M;i++){
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    return 0;
}