#include <stdio.h> 		 
#include <malloc.h>
#include <time.h> 
#include <mpi.h>  		
#include <cuda.h>

using namespace std;
    //--	This MPI+CUDA program generates n numbers in node Rank 0, sends them to node Rank 1.  
    //--	Then, Node 1 receives the data and sends them to it's GPU. On the GPU, all the numbers increase by 1
    //--	Finally, the results array returns to node rank 1
	//--	Note 1: All the required time for H2D, Kernel execution, and D2H are considered.
	//--	Note 2: For having MPI_Send time: [(Total time - (H2D time + Kernel time + D2H time)) / 2]
	//--	TO COMPILE	nvcc -arch=compute_37  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr//lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi mpicuda.cu -o program
    //--	TO RUN		!mpiexec -n 2 ./program xx yy zz
    //--	xx (integer) No.of input to generate
    //--	yy (integer) Range of input data to be generated randomly
	//--	zz (integer) number of iterations


// declare the kernel function
__global__ void add(int *A, int N)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<N) A[i] = A[i] + 1; 
}

    int main(int argc, char **argv){
		int i, k, myid, procs, n, range, err, maxiter;
		int iter = 0;
		int block_size, grid_size;
		MPI_Status status;
		double t_start = 0.0, t_end = 0.0;
      // initialize MPI_Init
		err = MPI_Init(&argc, &argv);
		if (err != MPI_SUCCESS){
			printf("\nError initializing MPI.\n");
			MPI_Abort(MPI_COMM_WORLD, err);
		} // end if

      // Get No. of processors
		MPI_Comm_size(MPI_COMM_WORLD, &procs);

      // Get processor id
		MPI_Comm_rank(MPI_COMM_WORLD, &myid);

		if (myid == 0) {// to print only once....
			if (argc < 4) {
			  printf("\n\tOOOPS...., INVALID No of Arguements,\n");
	      }
		} // end myid == 0

		if (argc < 4) {MPI_Finalize(); return 0;} // end if

		n     = atoi(argv[1]);  // get n
		range = atoi(argv[2]);  
		maxiter  = atoi(argv[3]);  
      
		int nBytes = n * sizeof(int);
	    if (myid == 0) cout<<"The size of data: "<< nBytes << " bytes\n";
		k = n; // No. of elements to be computed by each Processor
		
		while (iter < maxiter)
		{  
			iter++;
			if (myid == 0) {
				int *arr;
				int *recarr;
			
				//srand(time(NULL));
				time_t t1;
				time(&t1); // get system time
				srand(t1); // Initilize Random Seed
				
				// allocate space to generate data
				arr = new int[n * sizeof (int)];
				recarr = new int[n * sizeof (int)];
		  
				for(i = 0; i < n; i++)// generate random data
				arr[i] = rand() % range;
				
				t_start = MPI_Wtime();
				MPI_Send(arr, k, MPI_INT, 1, i, MPI_COMM_WORLD);
				MPI_Recv(recarr, k, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				t_end = MPI_Wtime();
				
				cout<<"Iteration: "<<iter<<" , Total Time: " << ((t_end - t_start)* 1e3) << "  (ms) \n"; 
				   
				// --free allocated spaces
				free (arr);//free allocated space for array a
				free (recarr);		   
			} // end myid == 0

			else{	//Node rank 1
				int *myarr = new int[k * sizeof (int)];
				int *sndmyarr = new int[k * sizeof (int)];
				float milliseconds_h2d, milliseconds_d2h, milliseconds_k = 0;
				int *a_d;
				
				block_size=32;
				grid_size = 512;
				dim3 dimBlock(block_size,1,1);
				dim3 dimGrid(grid_size,1,1);
				cudaEvent_t start_h2d, stop_h2d, start_d2h, stop_d2h, start_k, stop_k;
				cudaEventCreate(&start_h2d);
				cudaEventCreate(&start_d2h);
				cudaEventCreate(&start_k);
				cudaEventCreate(&stop_d2h);
				cudaEventCreate(&stop_h2d);
				cudaEventCreate(&stop_k);
				cudaMalloc((void **)&a_d,k*sizeof(int));
				
				//Receiving data from master (Rank=0)
				MPI_Recv(myarr, k, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
				
				//Sending data to the GPU
				cudaEventRecord(start_h2d);
				cudaMemcpy(a_d,myarr,k*sizeof(int),cudaMemcpyHostToDevice);  
				cudaEventRecord(stop_h2d);
				cudaEventSynchronize(stop_h2d);
				cudaEventElapsedTime(&milliseconds_h2d, start_h2d, stop_h2d);
		
				//Stating kernel
				cudaEventRecord(start_k);
				add<<<grid_size,block_size>>>(a_d,k);
				cudaDeviceSynchronize();
				cudaEventRecord(stop_k);
				cudaEventSynchronize(stop_k);
				cudaEventElapsedTime(&milliseconds_k, start_k, stop_k);

				//Returning data from device memory to the host
				cudaEventRecord(start_d2h);
				cudaMemcpy(sndmyarr,a_d,k*sizeof(int),cudaMemcpyDeviceToHost);
				cudaEventRecord(stop_d2h);
				cudaEventSynchronize(stop_d2h);
				cudaEventElapsedTime(&milliseconds_d2h, start_d2h, stop_d2h); 

				//Check if cuda operations are successful
				cudaError_t err = cudaGetLastError();
				if ( err != cudaSuccess )
					printf("CUDA Error: %s\n", cudaGetErrorString(err));

				printf("\nIteration: %d , H2D time: %f , Kernel time: %f , D2H time: %f   (ms) \n", iter, milliseconds_h2d, milliseconds_k, milliseconds_d2h);
				
				//Returning result to the master node
				MPI_Send(sndmyarr, k, MPI_INT, 0, myid, MPI_COMM_WORLD);   
				
				//free allocated spaces
				free(myarr);
				free(sndmyarr);
				cudaFree(a_d);
				cudaEventDestroy(start_h2d);
				cudaEventDestroy(stop_h2d);
				cudaEventDestroy(start_d2h);
				cudaEventDestroy(stop_d2h);
				cudaEventDestroy(start_k);
				cudaEventDestroy(stop_k);
			} // end else
		}
		 MPI_Finalize();
		 return 0;
	} // end main