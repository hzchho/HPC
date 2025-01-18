#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include<math.h>

int thread_count=8;
struct EQ{
    double a;
    double b;
    double c;
    double value[8];
    //-b,b^2,4ac,2a,det,√b^2-4ac,-b+,-b-
} eq;

int isok[8];

void* solve(void* rank){
    long my_rank=(long)rank;
    //busy-waiting
    while(isok[my_rank]!=1);

    switch(my_rank){
        case 0://-b
            eq.value[0]=-1*eq.b;
            break;
        case 1://b^2
            eq.value[1]=pow(eq.b,2);
            break;
        case 2://4ac
            eq.value[2]=4*eq.a*eq.c;
            break;
        case 3://2a
            eq.value[3]=2*eq.a;
            break;
        case 4://det
            eq.value[4]=eq.value[1]-eq.value[2];
            break;
        case 5://√det
            if(eq.value[4]<0){
                break;
            }
            eq.value[5]=sqrt(eq.value[4]);
            break;
        case 6://(-b-√det)/2a
            if(eq.value[4]<0){
                break;
            }
            eq.value[6]=(eq.value[0]-eq.value[5])/eq.value[3];
            break;
        case 7://(-b+√det)/2a
            if(eq.value[4]<0){
                break;
            }
            eq.value[7]=(eq.value[0]+eq.value[5])/eq.value[3];
            break;
    }
    isok[(my_rank+1)%thread_count]=1;
    return NULL;
}

int main(int argc, char* argv[]){
    long thread;
    pthread_t* thread_handles;
    printf("Please enter the coefficients of the quadratic equation.(a!=0):\n");
    scanf("%lf %lf %lf",&eq.a,&eq.b,&eq.c);

    // thread_count=strtol(argv[1], NULL, 10);
    thread_handles=malloc (thread_count*sizeof(pthread_t));
    
    for(int i=0;i<8;i++){
        isok[i]=0;
    }
    isok[0]=1;
    
    for(thread=0;thread<thread_count;thread++){
        pthread_create(&thread_handles[thread], NULL, solve, (void*)thread);
    }

    for(thread=0;thread<thread_count;thread++){
        pthread_join(thread_handles[thread], NULL);
    }
 
    printf("(%lf)x^2 + (%lf)x + (%lf) ",eq.a,eq.b,eq.c);
    if(eq.value[4]<0){
        printf("Has no solution!\n");
    }else if(eq.value[4]==0){
        printf("Has 1 solution: x= %.6lf\n",eq.value[6]);
    }else{
        printf("Has 2 solution: x1= %.6lf  x2= %.6lf\n",eq.value[6],eq.value[7]);
    }

    free(thread_handles);
    return 0;

}
