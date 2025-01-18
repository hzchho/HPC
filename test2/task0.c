#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>
#include<time.h>

int thread_count;
int m, n, k;
double **A;
double **B;
double **C;

void* Pth_mat_vect(void* rank){
    long my_rank=(long)rank;
    int local_m=m/thread_count;
    int begin_row=my_rank*local_m;
    int end_row=(my_rank+1)*local_m;
    for(int i=begin_row;i<end_row;i++){
        for(int j=0;j<k;j++){
            C[i][j]=0.0;
            for(int l=0;l<n;l++){
                C[i][j]+=A[i][l]*B[l][j];
            }
        }
    }
    return NULL;
}

int main(int argc, char* argv[]){
    printf("Input matrix scale:\n");
    scanf("%d %d %d",&m,&n,&k);
    long thread;
    pthread_t* thread_handles;
    
    thread_count=strtol(argv[1], NULL, 10);
    thread_handles=malloc(thread_count*sizeof(pthread_t));
    
    srand(time(NULL)); 
    A=malloc(m*sizeof(double*));
    B=malloc(n*sizeof(double*));
    C=malloc(m*sizeof(double*));
    for(int i=0;i<m;i++){
        A[i]=malloc(n*sizeof(double));
        C[i]=malloc(k*sizeof(double));
    }
    for(int i=0;i<n;i++){
        B[i]=malloc(k*sizeof(double));
    }
    //initialize
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            A[i][j]=rand()*5.53/(double)RAND_MAX;
            // A[i][j]=0.05;
        }
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
            B[i][j]=rand()*3.35/(double)RAND_MAX;
            // B[i][j]=0.05;
        }
    }

    clock_t start=clock();
    for(thread=0;thread<thread_count;thread++){
        pthread_create(&thread_handles[thread], NULL, Pth_mat_vect, (void*)thread);
    }

    for(thread=0;thread<thread_count;thread++){
        pthread_join(thread_handles[thread], NULL);
    }
    clock_t end=clock();
    double my_time=((double)(end-start))*1000.0/CLOCKS_PER_SEC;

    printf("Used time: %.4lf ms",my_time);
    //free
    for(int i=0;i<n;i++){
        free(A[i]);
    }
    free(A);
    for(int i=0;i<k;i++){
        free(B[i]);
        free(C[i]);
    }
    free(B);
    free(C);
    free(thread_handles);
}