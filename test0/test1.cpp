#include<iostream>
#include<cstdlib>
#include<ctime>
#include<chrono>
#include<iomanip>
#include<random>
using namespace std;

int main(){
	int M,N,K;
	printf("PUT IN NUM:\n"); 
	scanf("%d %d %d",&M,&N,&K); 
	
	double **A=new double*[M];
    double **B=new double*[N];
    double **C=new double*[M];
	for(int i=0;i<M;i++){
		A[i]=new double[N];
	}
    for(int i=0;i<N;i++){
    	B[i]=new double[K];
	}
	for(int i=0;i<M;i++){
		C[i]=new double[K];
	}
    
    
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<> dis(0.0,1.0); 
	
    for(int i=0;i<M;i++){
    	for(int j=0;j<N;j++){
    		A[i][j]=dis(gen);
		}
	}
	
	for(int i=0;i<N;i++){
    	for(int j=0;j<K;j++){
    		B[i][j]=dis(gen);
		}
	}
    
    for(int i=0;i<M;i++){
    	for(int j=0;j<K;j++){
    		C[i][j]=0.0;
		}
	}

    auto start=chrono::high_resolution_clock::now();
    
    for(int i=0;i<M;i++){
    	for(int j=0;j<K;j++){
    		for(int l=0;l<N;l++){
    			C[i][j]+=A[i][l]*B[l][j];
			}
		}
	}

    auto end=chrono::high_resolution_clock::now();
    chrono::duration<double> time_=end-start;
    
    printf("Used time: %.6f s",time_.count());

	for(int i=0;i<M;i++){
		delete []A[i];
	}
    for(int i=0;i<N;i++){
    	delete []B[i];
	}
	for(int i=0;i<M;i++){
		delete []C[i];
	} 
	delete A;
	delete B;
	delete C;
}
