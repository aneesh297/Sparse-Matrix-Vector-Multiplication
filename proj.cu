#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utilities.h"

using namespace std;



__global__ void spmv(float *values, int *col_idx, int *row_off,float * vect,\
 float res[], int m, int n, int *bin, int bin_size,int bin_row_len, int nnz)
{
	int tid = threadIdx.x;
	int lid = tid%32;
	int vid = tid/32;
	float sum = 0;
	int row = bin[lid];
	int row_idx = row_off[row];
	int next_row_idx;
	if(row < (m-1))
		next_row_idx = row_off[row+1];
	else
		next_row_idx = nnz;

		printf("\nblockid = %d, threadid = %d,row = %d, row_idx = %d, next_row_idx = %d \n",blockIdx.x, tid, row, row_idx, next_row_idx);

	for(int i = row_idx + vid; i < next_row_idx; i+= 1<<(bin_row_len - 1))
	{
		sum += values[i] * vect[col_idx[i]];
		printf("\nblockid = %d, threadid = %d,value = %f, vect = %f \n",blockIdx.x, tid, values[i], vect[col_idx[i]]);
	} 

	for(int i = bin_row_len; i > 0; i--)
		sum += __shfl_down(sum,i);

	

	if(vid == 0)
		res[row] += sum;
	
}

int main()
{
	srand (time(NULL)); //Set current time as random seed.
	int m = 5, n = 5; //Matrix dimensions
	int nnz = 0, nnz_row[m], nnz_max = 0; // nnz -> number of non zeros
	float *mat[m], *vect, *res;
	float *values;
	int *col_idx, *row_off;



	for(int i = 0; i < m; i++)
	{
		mat[i] = sparse_gen(n, nnz, nnz_row[i], nnz_max);
	}

	cout<<"\nMatrix generated: \n";
	display_matrix(mat, m , n);

	vect = vect_gen(n);
	cout<<"\nVector generated: \n";
	display_vector(vect, n);

	cout<<"NNZ: "<<nnz<<endl;

	values = new float [nnz];
	col_idx = new int[nnz];
	row_off = new int[m];

	vector <vector<int> > bins(nnz_max+1);


	to_csr(mat, values, col_idx, row_off, m, n);
	display_csr(values, col_idx, row_off, nnz, m);


	res = new float[n];

	simple_spmv(res, vect, values, col_idx, row_off, nnz, m, n);

	cout<<"Result vector: \n";
	display_vector(res, n);

	calculate_bin_size(bins, nnz_row, m);

	for (int i = 0; i < nnz_max+1; ++i)
	{

		cout<<"Bin Size: "<<i<<endl;
		for(int j = 0; j < bins[i].size(); j++)
		{
			cout<<bins[i][j]<<" ";
		}

		cout<<endl;
	}



	// CUDA stuff

	int *dcol_idx, *drow_off, *dbin;
	float *dvect, *dres, *dvalues;

	//Events are used to time operations
	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



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


	//ACSR Binning
	for(int i = 1; i < bins.size(); i++)
	{
		if(bins[i].size()>0)
		{
			cout<<"Currently Bin "<<i<<endl;
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
		}
		
	}

	//Copy results into main memory
	cudaEventRecord(start);
	cudaMemcpy(res, dres, (n)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cout<<"Output Vector: ";

	display_vector(res,n);

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