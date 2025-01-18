#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>

int M=1024,N=1024,K=1024;
int thread_count;

void cal_mat(double** A,double** B, double **C, int op){
    int i=0,j=0,l=0;
    if(op==1){
        #pragma omp parallel for num_threads(thread_count) shared(A,B,C,M,N,K) private(i,j,l)
        for(i=0;i<M;i++){
            for(j=0;j<K;j++){
                C[i][j]=0.0;
                for(l=0;l<N;l++){
                    C[i][j]+=A[i][l]*B[l][j];
                }
            }
        }
    }else if(op==2){
        #pragma omp parallel for num_threads(thread_count) shared(A,B,C,M,N,K) private(i,j,l) schedule(static,16)
        for(i=0;i<M;i++){
            for(j=0;j<K;j++){
                C[i][j]=0.0;
                for(l=0;l<N;l++){
                    C[i][j]+=A[i][l]*B[l][j];
                }
            }
        }
    }else if(op==3){
        #pragma omp parallel for num_threads(thread_count) shared(A,B,C,M,N,K) private(i,j,l) schedule(dynamic,16)
        for(i=0;i<M;i++){
            for(j=0;j<K;j++){
                C[i][j]=0.0;
                for(l=0;l<N;l++){
                    C[i][j]+=A[i][l]*B[l][j];
                }
            }
        }
    }else{
        #pragma omp parallel for num_threads(thread_count) shared(A,B,C,M,N,K) private(i,j,l) schedule(guided,32)
        for(i=0;i<M;i++){
            for(j=0;j<K;j++){
                C[i][j]=0.0;
                for(l=0;l<N;l++){
                    C[i][j]+=A[i][l]*B[l][j];
                }
            }
        }
    }
}

int main(int argc, char* argv[]){
    thread_count=strtol(argv[1], NULL, 10);
    
    double** A=malloc(sizeof(double*)*M);
    double** B=malloc(sizeof(double*)*N);
    double** C1=malloc(sizeof(double*)*M);
    double** C2=malloc(sizeof(double*)*M);
    double** C3=malloc(sizeof(double*)*M);
    double** C4=malloc(sizeof(double*)*M);

    for(int i=0;i<M;i++){
        A[i]=malloc(sizeof(double)*N);
    }
    for(int i=0;i<N;i++){
        B[i]=malloc(sizeof(double)*K);
    }
    for(int i=0;i<M;i++){
        C1[i]=malloc(sizeof(double)*K);
        C2[i]=malloc(sizeof(double)*K);
        C3[i]=malloc(sizeof(double)*K);
        C4[i]=malloc(sizeof(double)*K);
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

    double time0=omp_get_wtime();
    cal_mat(A,B,C1,1);
    double time1=omp_get_wtime();
    cal_mat(A,B,C2,2);
    double time2=omp_get_wtime();
    cal_mat(A,B,C3,3);
    double time3=omp_get_wtime();
    cal_mat(A,B,C4,4);
    double time4=omp_get_wtime();
    
    printf("Default schedule:\nFirst_num: %.4lf Used time: %.4lfsec\n",C1[0][0],time1-time0);
    printf("Static schedule:\nFirst_num: %.4lf Used time: %.4lfsec\n",C2[0][0],time2-time1);
    printf("Dynamic schedule:\nFirst_num: %.4lf Used time: %.4lfsec\n",C3[0][0],time3-time2);
    printf("Guided schedule:\nFirst_num: %.4lf Used time: %.4lfsec\n",C4[0][0],time4-time3);
    
    for(int i=0;i<M;i++){
        free(A[i]);
    }
    for(int i=0;i<N;i++){
        free(B[i]);
    }
    for(int i=0;i<M;i++){
        free(C1[i]);
        free(C2[i]);
        free(C3[i]);
        free(C4[i]);
    }
    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C3);
    free(C4);
    return 0;
}