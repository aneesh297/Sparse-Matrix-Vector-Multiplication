#include <bits/stdc++.h>
#include "utilities.h"
#include "io.h"

using namespace std;

#define BIN_MAX 30
#define ROW_MAX 1024
#define THREAD_LOAD 5

__inline__ __device__
float warpReduceSum(float val) {
  val += __shfl_down(val, 16);
  val += __shfl_down(val, 8);
  val += __shfl_down(val, 4);
  val += __shfl_down(val, 2);
  val += __shfl_down(val, 1);
  return val;
}

__inline__ __device__
float blockReduceSum(float val) {

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / 32.0) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void spmv(const float * __restrict__ values,const int * __restrict__ col_idx,const int * __restrict__ row_off,cudaTextureObject_t vect,\
 float * __restrict__ res , int  m, int  n,const int * __restrict__  bin, int  bin_size,int  N, int nnz)
{
	int tid = threadIdx.x;
	float sum = 0;
	int row = bin[blockIdx.x];
	int row_idx = row_off[row];
	int next_row_idx;

	next_row_idx = row_off[row+1];


	for(int i = row_idx + tid; i < next_row_idx; i+= blockDim.x)
	{
		sum += values[i] * tex1Dfetch<float>(vect,col_idx[i]);//vect[col_idx[i]];
	}

	sum = blockReduceSum(sum);

	if(tid == 0)
		res[row] = sum;

}
////////////////////////////////////////////////////////////////////////////////////
// Kernel for dynamic parallelism
// flag -rdc=true should be set if this kernel is called
////////////////////////////////////////////////////////////////////////////////////
/*
__global__ void row_specific_spmv(float* __restrict__ values,int * __restrict__ col_idx,int * __restrict__ row_off,\
 float * __restrict__ x,float * __restrict__ res,int m,int n,int nnz,int row,int noOfThreads){
  int tid = threadIdx.x;
  int lid = tid%32;
  int vid = tid/32;

  float sum = 0;

  int row_idx = row_off[row];
	int next_row_idx;
	if(row < (m-1))
		next_row_idx = row_off[row+1];
	else
		next_row_idx = nnz;
	for(int i = row_idx + tid; i < next_row_idx; i+= noOfThreads)
	{
		sum += values[i] * x[col_idx[i]];
	}

	__syncthreads();

	sum = blockReduceSum(sum);

	if(lid == 0 && vid == 0)
		res[row] = sum;

}
__global__ void dynamicParallelParent(float * __restrict__ values, int * __restrict__ col_idx, int * __restrict__ row_off,float * __restrict__ x,\
 float * __restrict__ res, int  m, int  n, int nnz,int * __restrict__ G1, int G1_size){
  int tid = threadIdx.x;
  // printf("threadIdx = %d\n",tid);
  int row = G1[tid];
  int row_idx = row_off[row];
  int next_row_idx;
  if(row==m-1){
    next_row_idx = nnz;
  }
  else
    next_row_idx = row_off[row+1];

  int NNZ = next_row_idx - row_idx;
  int bsize = (NNZ-1)/THREAD_LOAD + 1;


  row_specific_spmv<<<1,bsize>>>(values,col_idx,row_off,x,res,m,n,nnz,row,bsize);
 }
*/
 ////////////////////////////////////////////////////////////////////////////////////


int calc_bin_index(int nnz){
  if(nnz==0 | nnz==1)
    return nnz;

  int cnt = 0,orig_nnz = nnz;
  while(nnz>0){
    nnz>>=1;
    cnt++;
  }
  if(!(orig_nnz & (orig_nnz-1)))
    return cnt-1;
  else return cnt;

}


// Matrix : m x n
// Vector : n x 1
float* driver(float *values, int *col_idx, int* row_off, float* x, float* y, int m, int n, int nnz){
  int max_nnz = INT_MIN;
  for(int i=1;i<m;i++)
    max_nnz = max(max_nnz,row_off[i]-row_off[i-1]);
  max_nnz = max(max_nnz, m-row_off[m-1]);

  //Timer setup
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int max_bins = calc_bin_index(max_nnz);
   cout<<"max_bins = "<<max_bins<<"\n";
  vector<int> bins[max_bins+1];

  for(int i = 1;i<m;i++){
    int nnz = row_off[i]-row_off[i-1];
    int bin_index = calc_bin_index(nnz);
    bins[bin_index].push_back(i-1);
  }

  int last_nnz = nnz-row_off[m-1];
  bins[calc_bin_index(last_nnz)].push_back(m-1);

  for(int i=0;i<=max_bins;i++){
    cout<<i<<"-->"<<bins[i].size();
    cout<<"\n";
  }

  int *dcol_idx, *drow_off;
  float *dvect, *dres, *dvalues;

  //Memory Allocation
  cout<<"Allocating memory\n";
  cudaEventRecord(start);
  cudaMalloc((void**)&dcol_idx, (nnz)*sizeof(int));
  cudaMalloc((void**)&drow_off, (m+1)*sizeof(int));
  cudaMalloc((void**)&dvect, (n)*sizeof(float));
  cudaMalloc((void**)&dres, (m)*sizeof(float));
  cudaMalloc((void**)&dvalues, (nnz)*sizeof(float));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout<<"Memory Allocation successful: "<<milliseconds<<" ms\n";


  //Copying memory to GPU
  cout<<"Copying memory to GPU\n";
  cudaEventRecord(start);
  cudaMemcpy(dcol_idx, col_idx, (nnz)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(drow_off, row_off, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dvect, x, (n)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvalues, values, (nnz)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dres, 0, n * sizeof(float));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout<<"Memory copy complete: "<<milliseconds<<"ms\n";

  float kernel_time = 0;

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = dvect;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32; // bits per channel
  resDesc.res.linear.sizeInBytes = n*sizeof(float);

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  // create texture object: we only have to do this once!
  cudaTextureObject_t tdvect=0;
  cudaCreateTextureObject(&tdvect, &resDesc, &texDesc, NULL);

  //Calculate G2
  for(int i = 1; i <=min(max_bins,BIN_MAX); i++)
	{
		if(bins[i].size()>0)
		{
      cout<<"Currently Bin "<<i<<endl;
      int *dbin;
			cudaMalloc((void**)&dbin, bins[i].size() * sizeof(int));

      int arr[bins[i].size()]; //Temporary array to store a single bin
			for(int j = 0; j < bins[i].size();j++)
				arr[j] = bins[i][j];

			cudaMemcpy(dbin, arr, (bins[i].size())*sizeof(int), cudaMemcpyHostToDevice);

			int dimBlock = (1 << (i - 1));
      if(dimBlock > 1024)
        dimBlock = 1024;
			dim3 dimGrid(bins[i].size());
      cout<<"Total No of threads: "<<dimBlock*bins[i].size()<<endl;

      cout<<"Executing Kernel: \n";
      cudaEventRecord(start);
			spmv<<<dimGrid,dimBlock>>>(dvalues, dcol_idx, drow_off, tdvect, dres, m, n, dbin, bins[i].size(), i, nnz);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      cout<<"Bin "<<i<<" execution complete: "<<milliseconds<<"ms\n";

			cudaFree(dbin);

      kernel_time += milliseconds;
		}
	}

  printf("\n\nGPU time taken for G2: %f ms\n\n", kernel_time);

  int *G1,*dG1;
  G1 = (int*)malloc(sizeof(int)*(m));
  int no_of_bigrows = 0;
  for(int i = BIN_MAX+1;i<=max_bins;i++){
    for(int j=0;j<bins[i].size();j++){
      G1[no_of_bigrows++] = bins[i][j];
    }
  }

  cout<<"no_of_bigrows = "<<no_of_bigrows<<"\n";
  cout<<"\n\n";
  cudaMalloc((void**)&dG1,(no_of_bigrows)*sizeof(int));
  cudaMemcpy(dG1,G1,no_of_bigrows*sizeof(int),cudaMemcpyHostToDevice);

  cout<<"Executing G1 Kernel: \n";
  cudaEventRecord(start);
  //dynamicParallelParent<<<1,no_of_bigrows>>>(dvalues, dcol_idx, drow_off, dvect, dres, m, n, nnz, dG1, no_of_bigrows);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout<<"Time taken for G1: "<<milliseconds<<" ms\n";

  cout<<"Total GPU time = "<<kernel_time + milliseconds<<"\n";

  float* kres = (float*)malloc(m*sizeof(float));
  cudaMemcpy(kres, dres, (m)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDestroyTextureObject(tdvect);
  return kres;
}

int main(){
   int n,m,nnz=0;
   int nnz_max;
   float *x;
   srand (time(NULL)); //Set current time as random seed.

  conv(nnz, m, n, nnz_max);
  x = vect_gen(n);
  float* y = (float*) malloc(m*sizeof(float));
  float *res = new float[m];

  clock_t begin = clock();
  simple_spmv(res, x, values, col_idx, row_off, nnz, m, n);
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout<<"\nTime taken for sequential: "<<elapsed_secs*1000<<" ms\n\n\n";

   y = driver(values,col_idx,row_off,x,y,m,n,nnz);
   checker(y,res,m);

   cout<<"\n\n";
}
