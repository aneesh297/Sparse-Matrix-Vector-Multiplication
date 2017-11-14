#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "io_cusparse.h"
#include "utilities_cusparse.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
/********/
/* MAIN */
/********/
int main()
{
	// --- Initialize cuSPARSE

  cudaEvent_t start,stop;
  float milliseconds = 0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	cusparseHandle_t handle;	cusparseCreate(&handle);
  cusparseMatDescr_t descrA;		cusparseCreateMatDescr(&descrA);
	cusparseSetMatType		(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

   int n,m,nnz=0;
   int nnz_max;
   double *x;

   conv(nnz, m, n , nnz_max);
   x = vect_gen(n);


	double *d_A;			gpuErrchk(cudaMalloc(&d_A, nnz * sizeof(double)));
	int *d_A_RowIndices;	gpuErrchk(cudaMalloc(&d_A_RowIndices, (m + 1) * sizeof(int)));
	int *d_A_ColIndices;	gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz * sizeof(int)));
  double* d_x_dense; gpuErrchk(cudaMalloc(&d_x_dense, n * sizeof(double)));
  double* d_y_dense; gpuErrchk(cudaMalloc(&d_y_dense, m * sizeof(double)));

	gpuErrchk(cudaMemcpy(d_A, values, nnz * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A_RowIndices, row_off, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_A_ColIndices, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_x_dense, x, n * sizeof(double), cudaMemcpyHostToDevice));
  cudaMemset(d_y_dense,0,m*sizeof(double));


	const double alpha = 1.;
	const double beta  = 0.;

  clock_t begin_cusparse = clock();
	cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_x_dense,
	                            &beta, d_y_dense);
  cudaDeviceSynchronize();
  clock_t end_cusparse = clock();
  double elapsed_secs_cusparse = double(end_cusparse - begin_cusparse) / CLOCKS_PER_SEC;
  cout<<"\nTime taken for cusparse: "<<elapsed_secs_cusparse*1000<<" ms\n\n\n";


  double* h_y_dense; h_y_dense = (double*)malloc(m * sizeof(double));
  gpuErrchk(cudaMemcpy(h_y_dense, d_y_dense, m * sizeof(double), cudaMemcpyDeviceToHost));

  double *res = new double[m];
  clock_t begin = clock();
  simple_spmv(res,x,values,col_idx,row_off,nnz, m, n);
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout<<"\nTime taken for sequential: "<<elapsed_secs*1000<<" ms\n\n\n";

  checker(h_y_dense,res,m);

  cout<<"\n\n";
}
