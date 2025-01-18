#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>
#include"parallel_fun.h"

int M,N,K;
double** A;
double** B;
double** C;

struct for_index{
	int start;
	int end;
	int increment;
	double **A;
	double **B;
	double **C;
	int scale1;
	int scale2;
};

void* functor(void* args){
    struct for_index *index=(struct for_index *)args;
	int n=index->scale1;
	int k=index->scale2;
    for(int i=index->start;i<index->end;i=i+index->increment){
        for(int j=0;j<n;j++){
			index->C[i][j]=0.0;
			for(int l=0;l<k;l++){
				index->C[i][j]+=index->A[i][l]*index->B[l][j];
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
    struct for_index* index=malloc(sizeof(struct for_index));
    index->A=A;
    index->B=B;
    index->C=C;
    index->scale1=N;
    index->scale2=K;
    
    double start=omp_get_wtime();
    parallel_for(0, M, 1, functor, (void*)index, thread_count);
    double end=omp_get_wtime();
    
    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            printf("%.2lf ",C[i][j]);
        }
        printf("\n");
    }

    printf("Used time: %.4lf sec",end-start);
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
    free(index);
    return 0;
}