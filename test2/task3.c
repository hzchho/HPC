#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>
#include<time.h>

int thread_count;
//y=x^2与x轴面积,x∈[0,1]，积分运算结果为1/3
int point_num=0;
int MAX_POINT_NUM=100000000;
int under=0;//在函数曲线下方的点数
pthread_mutex_t mutex;

double f(double x){
    return x*x;
}

void* set_point(void* rank){
    long my_rank=(long)rank;
    
    srand(time(NULL));
    while(1){
        pthread_mutex_lock(&mutex);
        if(point_num>=MAX_POINT_NUM){
            pthread_mutex_unlock(&mutex);
            return NULL;
        }
        double x=rand()*1.0/RAND_MAX;
        double y=rand()*1.0/RAND_MAX;
        // printf("(%.2lf,%.2lf)\n",x,y);
        if(y<f(x)){
            under+=1;
        }
        point_num+=1;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main(int argc, char* argv[]){
    long thread;
    pthread_t* thread_handles;
    pthread_mutex_init(&mutex, NULL);

    thread_count=strtol(argv[1], NULL, 10);
    thread_handles=malloc (thread_count*sizeof(pthread_t));
    
    for(thread=0;thread<thread_count;thread++){
        pthread_create(&thread_handles[thread], NULL, set_point, (void*)thread);
    }

    for(thread=0;thread<thread_count;thread++){
        pthread_join(thread_handles[thread], NULL);
    }


    double area=under*1.0/MAX_POINT_NUM;
    printf("Used Point: %d\n",MAX_POINT_NUM);
    printf("Used Pthread: %d\n",thread_count);
    printf("Area: %.6lf\n",area);

    pthread_mutex_destroy(&mutex);
    free(thread_handles);
    return 0;
}