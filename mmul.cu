#include <cuda.runtime.h>
#includ <stdio.h>

//m x n * n x p
//c[i][j] = sum(a[i][0:n] * b[0:n][j])

#define BLOCK_WIDTH 16
#define TILE_WIDTH 16

__global__ void gmem_kernel_mmul(float* d_a, float* d_b, float* d_c, int m, int n, in p) {
	int row = blockIdx.y * blockDim.y + threadIdy.y; // x index of d_c
	int col = blockIdx.x * blockDim.x + threadIdx.x; // y index of d_c

	float sum = 0;

	// have many overlapped item used for calculation for threads in block
	// improve by shared memory for locality
	if (row < m && col < p) {
		for (int i = 0; i < n; i++) {
			sum += d_a[row * n + i] * d_b[i * p + col];
		}
	}

	d_c[row * p + col] = sum;
}

__global__ void smem_kernel_mmul(float* d_a, float* d_b, float* d_c, int m, int n, int p) {
	int row = blockIdx.y * blockDim.y + threadIdy.y; // x index of d_c
	int col = blockIdx.x * blockDim.x + threadIdx.x; // y index of d_c

	//compute capability 1.x share mem capacity: 16kB 
	//compute capability 2.x share mem capacity: 48kB

	//square tile
	__shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

	float sum = 0;

	//square matrix 
	for (int itr = 0; itr < gridDim.x / TILE_WIDTH; itr++) {
		//each thread copy 1 element from d_a matrix and 1 element from d_b matrix
		
		//copy horizontally from global memory to shared memory

		//row * n + threadId.x -> starting x index in d_a matrix

		tile_a[threadIdy.y][threadIdx.x] = d_a[row * n + threadIdx.x + itr * TILE_WIDTH];

		//copy vertically from global memory to shared memory

		//threadIdy.y * p + col -> starting y index in d_b matrix

		tile_b[threadIdy.y][threadIdx.x] = d_b[threadIdy.y * p + col + itr * TILE_WIDTH * p];

		__syncthreads();

		// sum over tile, row x cols
		for (int i = 0; i < TILE_WIDTH; i++) { 
			sum += tile_a[threadIdy.y][i] * tile_b[i][threadIdx.x];
		}

		__syncthreads();
	}

	d_c[row * p + col] = sum;

}

void cu_mmul(float* a, float* b, float* c, int M, int N, int P) {
	float* d_a, d_b, d_c;

	int d_a_size = sizeof(float) * M * N;
	int d_b_size = sizeof(float) * N * P;
	int d_c_size = sizeof(float) * M * P;

	dim3 block_size;
	block_size.x = BLOCK_WIDTH;
	block_size.y = BLOCK_WIDTH;

	dim3 grid_size;
	grid.x = (M + block_size.x - 1) / block_size;
	grid.y = (P + block_size.y - 1) / block_size;

	cudaMalloc((void** )&d_a, d_a_size);
	cudaMalloc((void** )&d_b, d_b_size);
	cudaMalloc((void** )&d_c, d_c_size);


	cudaMemcpy(d_a, a, d_a_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, d_b_size, cudaMemcpyHostToDevice);

	//global memory approach
	//gmem_kerenl_mmul<<grid, block>>(&d_a, &d_b, &d_c, M, N, P);

	//share memory approach
	smem_kerenl_mmul<<grid, block>>(&d_a, &d_b, &d_c, M, N, P);

	cudaMemcpy(c, d_c, d_c_size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}