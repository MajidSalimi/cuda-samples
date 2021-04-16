## MPI+CUDA
This program is a simple `MPI+CUDA` that works for only 2 nodes. MPI is used for inter-node communication, and CUDA is used for utilizing GPU on the remote node.
This program generates n numbers on the master node (Rank 0), sends them to node Rank 1. As soon as node 1 receives the data, sends them to it's GPU. On the GPU, all the numbers increase by 1.
Finally, the results' array returns to the master node with MPI.

The program is designed to measure the communication time between nodes and within a node.

#### Note 1: 
To obtain a single `MPI_Send()` time (inter-node communication): [(Total time - (H2D time + Kernel time + D2H time)) / 2]

### How to Compile and Run?

TO COMPILE:
```
nvcc -arch=compute_37  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr//lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi mpicuda.cu -o program
```
TO RUN:	
```
mpiexec -n 2 ./program xx yy zz
```

- xx (integer): Number of random integer numbers to generate

- yy (integer): Range of input data to be generated randomly

- zz (integer): Number of iterations

  
 #### Note 2: 
 To achieve the required commandline MPI flags, you can run the following command:
 
 ```
 mpicc -showme 
 ```
