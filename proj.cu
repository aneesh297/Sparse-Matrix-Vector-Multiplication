#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utilities.h"

using namespace std;



__global__ void spmv(float *values, int *col_idx, int *row_off,float * vect, float * res, int m, int n, int *bin, int bin_size)
{

}

int main()
{
	srand (time(NULL));
	int m = 5, n = 5;
	int nnz = 0, nnz_row[m], nnz_max = 0; 
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
	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



	cout<<"Allocating memory\n";
	cudaEventRecord(start);
	cudaMalloc((void**)&dcol_idx, (nnz)*sizeof(int));
	cudaMalloc((void**)&drow_off, (m)*sizeof(int));
	cudaMalloc((void**)&dvect, (n)*sizeof(int));
	cudaMalloc((void**)&dres, (n)*sizeof(int));
	cudaMalloc((void**)&dvalues, (nnz)*sizeof(int));
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"Memory Allocation successful: "<<milliseconds<<"ms\n";


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