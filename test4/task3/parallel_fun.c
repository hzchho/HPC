#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include"parallel_fun.h"

struct for_index{
	int start;
	int end;
	int increment;
	double **A;
	double **B;
	double **C;
	int scale1;
	int scale2;
	double num1;
};

int min(int a, int b){
  	if(a>=b){
    	return a;
  	}
  	return b;
}

void parallel_for(int start, int end, int increment, void*(functor)(void*) ,void *arg , int num_threads){

    pthread_t* thread_handles=malloc(num_threads*sizeof(pthread_t));
	struct for_index* indice=malloc(num_threads*sizeof(struct for_index));
    
	struct for_index *data=(struct for_index *)arg;
    
	for(int i=0;i<num_threads;i++){
		indice[i].start=start+(end-start)/num_threads*i;
		indice[i].end=min(start+(end-start)/num_threads*(i+1),end);
		indice[i].increment=increment;
		indice[i].A=data->A;
		indice[i].B=data->B;
		indice[i].C=data->C;
		indice[i].scale1=data->scale1;
		indice[i].scale2=data->scale2;
		indice[i].num1=data->num1;

		pthread_create(&thread_handles[i], NULL, functor, &indice[i]);
	}

	for(int i=0;i<num_threads;i++){
		pthread_join(thread_handles[i], NULL);
	}
    
	free(thread_handles);
	free(indice);
}
