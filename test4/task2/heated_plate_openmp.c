# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <mpi.h>
# include <omp.h>


int main ( int argc, char *argv[] );

/******************************************************************************/

int main ( int argc, char *argv[] )
{
# define M 500
# define N 500
  
  int comm_sz;
  int my_rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  double diff;
  double epsilon = 0.001;
  int i;
  int iterations;
  int iterations_print;
  int j;
  double mean;
  double my_diff;
  double* u=malloc(sizeof(double)*M*N);
  double* local_u=malloc(sizeof(double)*(M/comm_sz)*N);
  double* w=malloc(sizeof(double)*M*N);
  double* local_w=malloc(sizeof(double)*(M/comm_sz)*N);

  double wtime;
  if(my_rank==0){
    printf ( "\n" );
    printf ( "HEATED_PLATE_OPENMP\n" );
    printf ( "  C/OpenMP version\n" );
    printf ( "  A program to solve for the steady state temperature distribution\n" );
    printf ( "  over a rectangular plate.\n" );
    printf ( "\n" );
    printf ( "  Spatial grid of %d by %d points.\n", M, N );
    printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon ); 
    printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
    printf ( "  Number of processes =            %d\n", comm_sz );
  }
/*
  Set the boundary values, which don't change. 
*/
  mean = 0.0;
  if(my_rank==0){
    for ( i = 1; i < M - 1; i++ )
    {
      w[i*N] = 100.0;
    }

    for ( i = 1; i < M - 1; i++ )
    {
      w[i*N+N-1] = 100.0;
    }

    for ( j = 0; j < N; j++ )
    {
      w[(M-1)*N+j] = 100.0;
    }

    for ( j = 0; j < N; j++ )
    {
      w[j] = 0.0;
    }

    for ( i = 1; i < M - 1; i++ )
    {
      mean = mean + w[i*N] + w[i*N+N-1];
    }

    for ( j = 0; j < N; j++ )
    {
      mean = mean + w[(M-1)*N+j] + w[j];
    }

    mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
    printf ( "\n" );
    printf ( "  MEAN = %f\n", mean );
  }
  MPI_Barrier(MPI_COMM_WORLD);
/* 
  Initialize the interior solution to the mean value.
*/
  MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(w, (M/comm_sz)*N, MPI_DOUBLE, local_w, (M/comm_sz)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  fuctor1(my_rank, comm_sz, M/comm_sz, N, local_w, mean);
  MPI_Gather(local_w, (M/comm_sz)*N, MPI_DOUBLE, w, (M/comm_sz)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
/*
  iterate until the  new solution W differs from the old solution U
  by no more than EPSILON.
*/
  iterations = 0;
  iterations_print = 1;
  if(my_rank==0){
    printf ( "\n" );
    printf ( " Iteration  Change\n" );
    printf ( "\n" );
  }
  wtime = MPI_Wtime();

  diff = epsilon;

  while ( epsilon <= diff )
  {
/*
  Save the old solution in U.
*/
  MPI_Bcast(w, M*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(u, (M/comm_sz)*N, MPI_DOUBLE, local_u, (M/comm_sz)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  fuctor2(my_rank, M/comm_sz, N, local_u, w);
  MPI_Gather(local_u, (M/comm_sz)*N, MPI_DOUBLE, u, (M/comm_sz)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

/*
  Determine the new estimate of the solution at the interior points.
  The new solution W is the average of north, south, east and west neighbors.
*/
  MPI_Bcast(u, M*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(w, (M/comm_sz)*N, MPI_DOUBLE, local_w, (M/comm_sz)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  fuctor3(my_rank, comm_sz, M/comm_sz, N, local_w, u);
  MPI_Gather(local_w, (M/comm_sz)*N, MPI_DOUBLE, w, (M/comm_sz)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

/*
  C and C++ cannot compute a maximum as a reduction operation.

  Therefore, we define a private variable MY_DIFF for each thread.
  Once they have all computed their values, we use a CRITICAL section
  to update DIFF.
*/
  diff=0.0;
  my_diff=0.0;
  MPI_Scatter(w, (M/comm_sz)*N, MPI_DOUBLE, local_w, (M/comm_sz)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(u, (M/comm_sz)*N, MPI_DOUBLE, local_u, (M/comm_sz)*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  my_diff=fuctor4(my_rank, comm_sz, M/comm_sz, N, local_w, local_u, my_diff);
  MPI_Reduce(&my_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    iterations++;
    if ( iterations == iterations_print )
    {
      if(my_rank==0){
        printf ( "  %8d  %f\n", iterations, diff );
      }

      iterations_print = 2 * iterations_print;
    }
  MPI_Barrier(MPI_COMM_WORLD);
  } 
  
  wtime = MPI_Wtime() - wtime;

  if(my_rank==0){
    printf ( "\n" );
    printf ( "  %8d  %f\n", iterations, diff );
    printf ( "\n" );
    printf ( "  Error tolerance achieved.\n" );
    printf ( "  Wallclock time = %f\n", wtime );
    printf ( "\n" );
    printf ( "HEATED_PLATE_OPENMP:\n" );
    printf ( "  Normal end of execution.\n" );
  }
  
  free(u);
  free(w);
  free(local_u);
  free(local_w);

  MPI_Finalize();
  return 0;

# undef M
# undef N
}


