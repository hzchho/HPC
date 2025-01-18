# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include <pthread.h>
# include "parallel_fun.h"

int main ( int argc, char *argv[] );

pthread_mutex_t mutex;

struct for_index{
	  int start;
	  int end;
	  int increment;
	  double **A;
	  double **B;
	  double **C;
	  int scale1;
	  int scale2;
    double mean;
};

void* functor1(void* arg){
    struct for_index *index=(struct for_index *)arg;
    int n=index->scale2;
    for(int i=index->start;i<index->end;i+=index->increment){
        for(int j=1;j<n-1;j++){
            index->B[i][j]=index->mean;
        }
    }

    return NULL;
}

void* functor2(void* arg){
    struct for_index *index=(struct for_index *)arg;
    int n=index->scale2;
    for(int i=index->start;i<index->end;i+=index->increment){
        for(int j=0;j<n;j++){
            index->A[i][j]=index->B[i][j];
        }
    }

    return NULL;
}

void* functor3(void* arg){
    struct for_index *index=(struct for_index *)arg;
    int n=index->scale2;
    for(int i=index->start;i<index->end;i+=index->increment){
        for(int j=1;j<n-1;j++){
            index->B[i][j]=(index->A[i-1][j]+index->A[i+1][j]+index->A[i][j-1]+index->A[i][j+1])/4.0;
        }
    }

    return NULL;
}

void* functor4(void* arg){
    struct for_index *index=(struct for_index *)arg;
    int n=index->scale2;
    double my_diff=0.0;
    for(int i=index->start;i<index->end;i+=index->increment){
        for(int j=1;j<n-1;j++){
            if(my_diff<fabs(index->B[i][j]-index->A[i][j])){
                my_diff=fabs(index->B[i][j]-index->A[i][j]);
            }
        }
    }

    pthread_mutex_lock(&mutex);
    printf("Pid: %lu: \n",(unsigned long)pthread_self());
    if(index->C[0][0]<my_diff){
      index->C[0][0]=my_diff;
    }
    pthread_mutex_unlock(&mutex);

    return NULL;
}

/******************************************************************************/

int main ( int argc, char *argv[] )
{
# define M 500
# define N 500
  int num_threads=4;
  omp_set_num_threads(num_threads);
  double diff;
  double epsilon = 0.001;
  int i;
  int iterations;
  int iterations_print;
  int j;
  double mean;
  double **u=malloc(sizeof(double*)*M);
  double **w=malloc(sizeof(double*)*M);
  for(int x=0;x<M;x++){
    u[x]=malloc(sizeof(double)*N);
    w[x]=malloc(sizeof(double)*N);
  }
  double wtime;

  printf ( "\n" );
  printf ( "HEATED_PLATE_OPENMP\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "  A program to solve for the steady state temperature distribution\n" );
  printf ( "  over a rectangular plate.\n" );
  printf ( "\n" );
  printf ( "  Spatial grid of %d by %d points.\n", M, N );
  printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon ); 
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
/*
  Set the boundary values, which don't change. 
*/
  mean = 0.0;

#pragma omp parallel shared ( w ) private ( i, j )
  {
#pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][0] = 100.0;
    }
#pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][N-1] = 100.0;
    }
#pragma omp for
    for ( j = 0; j < N; j++ )
    {
      w[M-1][j] = 100.0;
    }
#pragma omp for
    for ( j = 0; j < N; j++ )
    {
      w[0][j] = 0.0;
    }
/*
  Average the boundary values, to come up with a reasonable
  initial value for the interior.
*/
#pragma omp for reduction ( + : mean )
    for ( i = 1; i < M - 1; i++ )
    {
      mean = mean + w[i][0] + w[i][N-1];
    }
#pragma omp for reduction ( + : mean )
    for ( j = 0; j < N; j++ )
    {
      mean = mean + w[M-1][j] + w[0][j];
    }
  }
/*
  OpenMP note:
  You cannot normalize MEAN inside the parallel region.  It
  only gets its correct value once you leave the parallel region.
  So we interrupt the parallel region, set MEAN, and go back in.
*/
  mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
  printf ( "\n" );
  printf ( "  MEAN = %f\n", mean );
/* 
  Initialize the interior solution to the mean value.
*/
  
  double **v=malloc(sizeof(double*));
  v[0]=malloc(sizeof(double));
  struct for_index* data=malloc(sizeof(struct for_index));
  data->A=u;
  data->B=w;
  data->C=v;
  data->scale1=M;
  data->scale2=N;
  data->mean=mean;
    
  parallel_for(1, M-1, 1, functor1, (void*)data, num_threads);
/*
  iterate until the  new solution W differs from the old solution U
  by no more than EPSILON.
*/
  iterations = 0;
  iterations_print = 1;
  printf ( "\n" );
  printf ( " Iteration  Change\n" );
  printf ( "\n" );
  wtime = omp_get_wtime ( );

  diff = epsilon;

  while ( epsilon <= diff )
  {
/*
  Save the old solution in U.
*/
  parallel_for(0, M, 1, functor2, (void*)data, num_threads);

/*
  Determine the new estimate of the solution at the interior points.
  The new solution W is the average of north, south, east and west neighbors.
*/
  parallel_for(1, M-1, 1, functor3, (void*)data, num_threads);

/*
  C and C++ cannot compute a maximum as a reduction operation.

  Therefore, we define a private variable MY_DIFF for each thread.
  Once they have all computed their values, we use a CRITICAL section
  to update DIFF.
*/
    diff = 0.0;
    data->C[0][0]=diff;
    parallel_for(1, M-1, 1, functor4, (void*)data, num_threads);

    diff=data->C[0][0];
    iterations++;
    if ( iterations == iterations_print )
    {
      printf ( "  %8d  %f\n", iterations, diff );
      iterations_print = 2 * iterations_print;
    }
  } 
  wtime = omp_get_wtime ( ) - wtime;

  printf ( "\n" );
  printf ( "  %8d  %f\n", iterations, diff );
  printf ( "\n" );
  printf ( "  Error tolerance achieved.\n" );
  printf ( "  Wallclock time = %f\n", wtime );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "HEATED_PLATE_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );
  
  pthread_mutex_destroy(&mutex);
  free(data);
  return 0;

# undef M
# undef N
}
