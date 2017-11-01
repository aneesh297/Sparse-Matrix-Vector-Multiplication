#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utilities.h"

using namespace std;

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    	val += __shfl_down(val, offset);
  return val;
}
__global__ void spmv(float * __restrict__ values, int * __restrict__ col_idx, int * __restrict__ row_off,float * __restrict__ vect,\
 float res[], int  m, int  n, int *  bin, int  bin_size,int  N, int nnz)
{
	int tid = threadIdx.x;
	//__shared__ float rowsum;// = 0;
	//rowsum = 0;
	static __shared__ float shared[32];
	int lid = tid%32;
	int vid = tid/32;
	float sum = 0;
	int row = bin[blockIdx.x];
	int row_idx = row_off[row];
	int next_row_idx;
	if(row < (m-1))
		next_row_idx = row_off[row+1];
	else
		next_row_idx = nnz;

		//printf("\nblockid = %d, threadid = %d,row = %d, row_idx = %d, next_row_idx = %d \n",blockIdx.x, tid, row, row_idx, next_row_idx);

	float var = 1<<(N-1);

	for(int i = row_idx + tid; i < next_row_idx; i+= var)
	{
		sum += values[i] * vect[col_idx[i]];
		//printf("\nblockid = %d, threadid = %d,value = %f, vect = %f \n",blockIdx.x, tid, values[i], vect[col_idx[i]]);
		//printf("\nmultiplication: %.0f x %.0f\n", values[i], vect[col_idx[i]]);
	} 

	//printf("sum1 = %f\n", sum);
	__syncthreads();

	
	//printf("sum1 = %f\n", sum);

	//for(int i = N; i > 0; i--)
		//atomicAdd(&rowsum,sum); //+= __shfl_down(sum,i);

	// for(int i = N/2; i > 0; i/=2)
	// 	if(__shfl_down(sum, i) != sum)
	// 		sum += __shfl_down(sum, i);

	sum = warpReduceSum(sum);

	if (lid == 0) shared[vid]=sum;

	__syncthreads();   

	sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lid] : 0;

	//if (vid == 0) sum = warpReduceSum(sum);

	//__syncthreads();

	//printf("sum2 = %f\n", sum);

	//if(lid == 0)
		//res[row] = rowsum;
		//atomicAdd(&res[row],sum);// += sum;
	if(vid == 0)
		atomicAdd(&res[row],sum);
	
}

int main()
{
	srand (time(NULL)); //Set current time as random seed.
	int m = 100, n = 100; //Matrix dimensions
	int nnz = 0, nnz_row[m], nnz_max = 0; // nnz -> number of non zeros
	float *mat[m], *vect, *res;
	float *values;
	int *col_idx, *row_off;

	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);




	for(int i = 0; i < m; i++)
	{
		mat[i] = sparse_gen(n, nnz, nnz_row[i], nnz_max);
	}

	// cout<<"\nMatrix generated: \n";
	// display_matrix(mat, m , n);

	vect = vect_gen(n);
	// cout<<"\nVector generated: \n";
	// display_vector(vect, n);

	 cout<<"NNZ: "<<nnz<<endl;

	values = new float [nnz];
	col_idx = new int[nnz];
	row_off = new int[m];

	vector <vector<int> > bins(nnz_max+1);


	to_csr(mat, values, col_idx, row_off, m, n);
	//display_csr(values, col_idx, row_off, nnz, m);


	res = new float[m];

	clock_t begin = clock();

  	simple_spmv(res, vect, values, col_idx, row_off, nnz, m, n);

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	

	cout<<"\nTime taken for sequential: "<<elapsed_secs*1000<<"\n\n\n";


	// cout<<"Result vector: \n\t\t";
	//display_vector(res, n);

	calculate_bin_size(bins, nnz_row, m);

	// for (int i = 0; i < nnz_max+1; ++i)
	// {

	// 	cout<<"Bin Size: "<<i<<endl;
	// 	for(int j = 0; j < bins[i].size(); j++)
	// 	{
	// 		cout<<bins[i][j]<<" ";
	// 	}

	// 	cout<<endl;
	// }



	// CUDA stuff

	int *dcol_idx, *drow_off, *dbin;
	float *dvect, *dres, *dvalues;

	//Events are used to time operations



	//Memory Allocation
	cout<<"Allocating memory\n";
	cudaEventRecord(start);
	cudaMalloc((void**)&dcol_idx, (nnz)*sizeof(int));
	cudaMalloc((void**)&drow_off, (m)*sizeof(int));
	cudaMalloc((void**)&dvect, (n)*sizeof(float));
	cudaMalloc((void**)&dres, (n)*sizeof(float));
	cudaMalloc((void**)&dvalues, (nnz)*sizeof(float));
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"Memory Allocation successful: "<<milliseconds<<"ms\n";


	//Copying memory to GPU
	cout<<"Copying memory to GPU\n";
	cudaEventRecord(start);
	cudaMemcpy(dcol_idx, col_idx, (nnz)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(drow_off, row_off, (m)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dvect, vect, (n)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dvalues, values, (nnz)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(dres, 0, n * sizeof(float));
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"Memory copy complete: "<<milliseconds<<"ms\n";

	float kernel_time = 0;

	//ACSR Binning
	for(int i = 1; i < bins.size(); i++)
	{
		if(bins[i].size()>0)
		{
			// cout<<"Currently Bin "<<i<<endl;
			cudaMalloc((void**)&dbin, bins[i].size() * sizeof(int));

			int arr[bins[i].size()]; //Temporary array to store a single bin
			for(int j = 0; j < bins[i].size();j++)
				arr[j] = bins[i][j];

			cudaMemcpy(dbin, arr, (bins[i].size())*sizeof(int), cudaMemcpyHostToDevice);

			int dimBlock = (1 << (i - 1)) ; //2^(i-1) threads per block. Not sure if this is correct
			cout<<"Total No of threads: "<<dimBlock*bins[i].size()<<endl;
			dim3 dimGrid(bins[i].size());


			cout<<"Executing Kernel: ";
			cudaEventRecord(start);
			spmv<<<dimGrid,dimBlock>>>(dvalues, dcol_idx, drow_off, dvect, dres, m, n, dbin, bins[i].size(), i, nnz);
			cudaEventRecord(stop);

			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			cout<<"Bin "<<i<<" execution complete: "<<milliseconds<<"ms\n";
			cudaFree(dbin);
			kernel_time += milliseconds;
		}
		
	}

	float *kres = new float[m];

	//Copy results into main memory
	cudaEventRecord(start);
	cudaMemcpy(kres, dres, (n)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// cout<<"Output Vector: \t\t";

	//display_vector(kres,n);

	checker(res, kres, m);
	cout<<'\n';

	printf("\n\nGPU time taken: %f\n\n", kernel_time);

	cout<<"Freeing memory\n";
	cudaEventRecord(start);
	cudaFree(dcol_idx);
	cudaFree(drow_off);
	cudaFree(dvect);
	cudaFree(dres);
	cudaFree(dvalues);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"Memory Freed: "<<milliseconds<<"ms\n";

}