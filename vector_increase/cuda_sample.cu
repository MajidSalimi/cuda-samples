#include <iostream>
#include <cuda.h>
using namespace std;

// declare the kernel function
__global__ void add(int *A, int N)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<N) A[i] = A[i] + 10; 
}

int main(int argc,char **argv)
{
   int n = atoi(argv[1]);
   int nBytes = n*sizeof(int);
   cout<<"The size of data: "<< nBytes << "\n";
   int block_size, grid_size;
   int *a, *b;
   a = (int *)malloc(nBytes);
   b = (int *)malloc(nBytes);
   
   int *a_d;
   block_size=256;
   grid_size = 1024;
   dim3 dimBlock(block_size,1,1);
   dim3 dimGrid(grid_size,1,1);
   
   for(int i=0;i<n;i++)
      {
	  a[i]=i;
      b[i]=0;
	  }
   
   cudaMalloc((void **)&a_d,n*sizeof(int));
   
   cout<<"Copying to device..\n";
   clock_t start_h2d=clock();
   cudaMemcpy(a_d,a,n*sizeof(int),cudaMemcpyHostToDevice);
   clock_t end_h2d=clock();   
	  
   cout<<"Starting kernel\n";
   clock_t start_d=clock(); 
   add<<<grid_size,block_size>>>(a_d,n);
   cudaDeviceSynchronize();
   clock_t end_d = clock();

   cout<<"Copying to host..\n";
   clock_t start_d2h=clock();
   cudaMemcpy(b,a_d,n*sizeof(int),cudaMemcpyDeviceToHost);
   clock_t end_d2h=clock();
   
   cudaError_t err = cudaGetLastError();
   if ( err != cudaSuccess )
		printf("CUDA Error: %s\n", cudaGetErrorString(err));       
   
   double time_a = (double)(end_h2d-start_h2d)/CLOCKS_PER_SEC;
   double time_b = (double)(end_d-start_d)/CLOCKS_PER_SEC;
   double time_c = (double)(end_d2h-start_d2h)/CLOCKS_PER_SEC;

   cout<<"\nH2D time: "<<time_a<<" Kernel time: "<<time_b<<" D2H time: "<<time_c<< endl;
   cudaFree(a_d);
   free(a);
   free(b);
   return 0;
}