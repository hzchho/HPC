#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>
#include<time.h>

int thread_count;
int global_index=0;
int a[1000];
int a_sum=0;
pthread_mutex_t mutex;

void* add(void* rank){
    long my_rank=(long)rank;
    
    while(1){
        //mutex
        pthread_mutex_lock(&mutex);
        if(global_index>=1000){
            pthread_mutex_unlock(&mutex);
            return NULL;
        }
        for(int i=global_index;i<global_index+10;i++){
            a_sum+=a[i];
        }
        global_index+=10;
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

    srand(time(NULL)); 
    //initialize
    for(int i=0;i<1000;i++){
        a[i]=rand()*100/RAND_MAX;
    }

    for(thread=0;thread<thread_count;thread++){
        pthread_create(&thread_handles[thread], NULL, add, (void*)thread);
    }

    for(thread=0;thread<thread_count;thread++){
        pthread_join(thread_handles[thread], NULL);
    }
    int sum=0;
    for(int i=0;i<1000;i++){
        sum+=a[i];
    }
    printf("Global index: %d\n",global_index);
    printf("Sum of arr is: %d\n",a_sum);
    printf("Read sum is: %d\n",sum);

    pthread_mutex_destroy(&mutex);
    free(thread_handles);
    return 0;
}