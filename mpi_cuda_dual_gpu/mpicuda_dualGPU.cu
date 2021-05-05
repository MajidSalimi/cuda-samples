#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <mpi.h>
#include <cuda.h>
#include <unistd.h>

using namespace std;
    	//--	This MPI+CUDA multi-GPU program generates n numbers in node Rank 0, process them on its GPU, then sends them to node Rank 1 to be processed. At the end, the final result returns back to master node.
	//--	TO COMPILE		nvcc -arch=compute_37  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr//lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi mpicudanormal.cu -o program
	//--	TO RUN			mpiexec -n 2 ./program xx yy zz
    	//--	xx (integer) 	No.of input to generate
    	//--	yy (integer) 	Range of input data to be generated randomly
	//--	zz (integer) 	number of iterations


// declare the kernel function
__global__ void add(int *A, int N)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<N) A[i] = A[i] + 1;
}

__global__ void add2(int *A, int N)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<N) A[i] = A[i] + 1;
}


	int main(int argc, char **argv){
		int  myid, procs, n, range, err, maxiter;
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
		if (myid == 0) printf("The size of data: %d\n", nBytes);
		//k = n; // No. of elements to be computed by each Processor

		while (iter < maxiter)
		{
			iter++;
			if (myid == 0) {
				int *a_d;
				int *arr, *myarr;
				block_size=32;
				grid_size = 512;
				//srand(time(NULL));
				time_t t1;
				time(&t1); // get system time
				srand(t1); // Initilize Random Seed
				dim3 dimBlock(block_size,1,1);
				dim3 dimGrid(grid_size,1,1);
				float milliseconds_h2d, milliseconds_d2h, milliseconds_k = 0;
				cudaEvent_t start_h2d, stop_h2d, start_d2h, stop_d2h, start_k, stop_k;
				cudaEventCreate(&start_h2d);
				cudaEventCreate(&start_d2h);
				cudaEventCreate(&start_k);
				cudaEventCreate(&stop_d2h);
				cudaEventCreate(&stop_h2d);
				cudaEventCreate(&stop_k);
				myarr = new int[n * sizeof (int)];
				// allocate space to generate data
				arr = new int[n * sizeof (int)];

				for(int j = 0; j < n; j++)// generate random data
					arr[j] = rand() % range;
				//cudaSetDevice(0);
				//cudaDeviceEnablePeerAccess( 1, 0 );

				cudaMalloc((void **)&a_d , n*sizeof(int));
				//Sending data to the GPU
				cudaEventRecord(start_h2d);
				cudaMemcpy(a_d, arr, n*sizeof(int), cudaMemcpyHostToDevice);
				cudaEventRecord(stop_h2d);
				cudaEventSynchronize(stop_h2d);
				cudaEventElapsedTime(&milliseconds_h2d, start_h2d, stop_h2d);

				//Stating kernel
				cudaEventRecord(start_k);
				add2<<<grid_size,block_size>>>(a_d,n);
				cudaDeviceSynchronize();
				cudaEventRecord(stop_k);
				cudaEventSynchronize(stop_k);
				cudaEventElapsedTime(&milliseconds_k, start_k, stop_k);

				//Returning data from device memory to the host
				cudaEventRecord(start_d2h);
				cudaMemcpy(myarr, a_d, n*sizeof(int), cudaMemcpyDeviceToHost);
				cudaEventRecord(stop_d2h);
				cudaEventSynchronize(stop_d2h);
				cudaEventElapsedTime(&milliseconds_d2h, start_d2h, stop_d2h); 

				//Check if cuda operations are successful
				cudaError_t err1 = cudaGetLastError();
				if (err1 != cudaSuccess)
					printf("CUDA Error: %s\n", cudaGetErrorString(err1));

				//Sending result to the master node
				t_start = MPI_Wtime();
				MPI_Send(myarr, n, MPI_INT, 1, 0, MPI_COMM_WORLD);
				MPI_Recv(myarr, n, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				t_end = MPI_Wtime();

				//for(int j=0 ; j<n ; j++)
				//	printf("%d  ", myarr[j]);

				printf("\nRank: %d , Iteration: %d , H2D: %f , D2H: %f ,Kernel: %f , Total Time: (ms) %f\n",myid, iter, milliseconds_h2d, milliseconds_d2h, milliseconds_k , ((t_end - t_start)* 1e3));

				// --free allocated spaces
				free (arr);//free allocated space for array a
				free(myarr);
				cudaFree(a_d);
				cudaEventDestroy(start_h2d);
				cudaEventDestroy(stop_h2d);
				cudaEventDestroy(start_d2h);
				cudaEventDestroy(stop_d2h);
				cudaEventDestroy(start_k);
				cudaEventDestroy(stop_k);

				sleep(1);//waits between iterations
			} // end myid == 0

			else{	//Node rank 1
				float milliseconds_k, milliseconds_h2d, milliseconds_d2h = 0;
				int *a_dd, *myarr;
				//cudaSetDevice(1);
				block_size=32;
				grid_size = 512;
				dim3 dimBlock(block_size,1,1);
				dim3 dimGrid(grid_size,1,1);
				cudaEvent_t start_k, stop_k, start_h2d, stop_h2d, start_d2h, stop_d2h;
				cudaEventCreate(&start_k);
				cudaEventCreate(&stop_k);
				cudaEventCreate(&stop_h2d);
				cudaEventCreate(&stop_d2h);
				cudaEventCreate(&start_h2d);
				cudaEventCreate(&start_d2h);
				cudaMalloc((void **)&a_dd , n*sizeof(int));
				myarr = new int[n * sizeof (int)];

				//cudaSetDevice(1);
				//cudaDeviceEnablePeerAccess( 0, 0 );

				//Receiving data from master (Rank=0)
				MPI_Recv(myarr, n, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

				//Sending data to the GPU
				cudaEventRecord(start_h2d);
				cudaMemcpy(a_dd, myarr, n*sizeof(int), cudaMemcpyHostToDevice);
				cudaEventRecord(stop_h2d);
				cudaEventSynchronize(stop_h2d);
				cudaEventElapsedTime(&milliseconds_h2d, start_h2d, stop_h2d);

				//Stating kernel
				cudaEventRecord(start_k);
				add<<<grid_size,block_size>>>(a_dd,n);
				cudaDeviceSynchronize();
				cudaEventRecord(stop_k);
				cudaEventSynchronize(stop_k);
				cudaEventElapsedTime(&milliseconds_k, start_k, stop_k);

				//Returning data from device memory to the host
				cudaEventRecord(start_d2h);
				cudaMemcpy(myarr, a_dd, n*sizeof(int), cudaMemcpyDeviceToHost);
				cudaEventRecord(stop_d2h);
				cudaEventSynchronize(stop_d2h);
				cudaEventElapsedTime(&milliseconds_d2h, start_d2h, stop_d2h);

				//Check if cuda operations are successful
				cudaError_t err = cudaGetLastError();
				if ( err != cudaSuccess )
					printf("CUDA Error: %s\n", cudaGetErrorString(err));

				printf("\nRank: %d , Iteration: %d , H2D: %f, D2H: %f , Kernel: %f  (ms) \n",myid, iter, milliseconds_h2d, milliseconds_d2h, milliseconds_k);

				//Returning result to the master node
				MPI_Send(myarr, n, MPI_INT, 0, myid, MPI_COMM_WORLD);

				//free allocated spaces
				cudaFree(a_dd);
				free(myarr);
				cudaEventDestroy(start_d2h);
				cudaEventDestroy(stop_d2h);
				cudaEventDestroy(stop_h2d);
				cudaEventDestroy(start_h2d);
				cudaEventDestroy(start_k);
				cudaEventDestroy(stop_k);
			} // end else
		}
		 MPI_Finalize();
		 return 0;
	} // end main
