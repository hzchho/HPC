#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include<time.h>

int N=1000000000;
int thread_count;

double add_arr(double* a,int op){
    double sum=0.0;
    if(op==1){
        #pragma omp parallel for num_threads(thread_count) reduction(+:sum)
        for(int i=0;i<N;i++){
            sum+=a[i];
        }
    }else if(op==2){
        #pragma omp parallel for num_threads(thread_count) reduction(+:sum) schedule(static,1000)
        for(int i=0;i<N;i++){
            sum+=a[i];
        }
    }else if(op==3){
        #pragma omp parallel for num_threads(thread_count) reduction(+:sum) schedule(dynamic,1000)
        for(int i=0;i<N;i++){
            sum+=a[i];
        }
    }else{
        #pragma omp parallel for num_threads(thread_count) reduction(+:sum) schedule(guided,1)
        for(int i=0;i<N;i++){
            sum+=a[i];
        }
    }
    return sum;
}

int main(int argc, char* argv[]){
    double* arr=malloc(N*sizeof(double));
    thread_count=strtol(argv[1], NULL, 10);
    
    srand(time(NULL));
    #pragma omp parallel for num_threads(thread_count)
    for(int i=0;i<N;i++){
        arr[i]=rand()*1.0/(double)RAND_MAX;
    }

    double time0=omp_get_wtime();
    double sum1=add_arr(arr,1);
    double time1=omp_get_wtime();
    double sum2=add_arr(arr,2);
    double time2=omp_get_wtime();
    double sum3=add_arr(arr,3);
    double time3=omp_get_wtime();
    double sum4=add_arr(arr,4);
    double time4=omp_get_wtime();
    
    printf("Default schedule:\nSum: %.4lf Used time: %.4lfsec\n",sum1,time1-time0);
    printf("Static schedule:\nSum: %.4lf Used time: %.4lfsec\n",sum2,time2-time1);
    printf("Dynamic schedule:\nSum: %.4lf Used time: %.4lfsec\n",sum3,time3-time2);
    printf("Guided schedule:\nSum: %.4lf Used time: %.4lfsec\n",sum4,time4-time3);
    
    free(arr);
    return 0;
}