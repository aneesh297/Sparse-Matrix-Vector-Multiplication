#include "io.h"
#include "utilities.h"

#define BLOCK_SIZE 1024

// Parallel SpMV with one Thread per Row
__global__
void parallel_spmv_1(float * values, int * col_idx, int * row_off, float * vect, float * res, 
    int m, int n, int nnz){

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<m){
        int begin_index = row_off[row];
        int end_index = row_off[row+1];

        float row_sum = 0.0;
        for(int i = begin_index; i < end_index; i++){
            row_sum += (values[i] * vect[col_idx[i]]);
        }

        res[row] = row_sum;
    }

}
////////////////////////////


// Parallel SpMV with one Warp per Row
__global__
void parallel_spmv_2(float * values, int * col_idx, int * row_off, float * vect, float * res, 
    int m, int n, int nnz){

}
////////////////////////////


int main(){

    // Create Cuda Events //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    ////////////////////////////

    // Reading Dataset //
    int m,n,nnz,nnz_max;

    conv(nnz,m,n,nnz_max,false);  // Defined in io.h

    cout<<"\nrows    = "<<m;
    cout<<"\ncolumns = "<<n;
    cout<<"\nnnz     = "<<nnz;
    cout<<"\nnnz_max = "<<nnz_max;
    cout<<"\n\n";

    float *vect = vect_gen(n,false); //generating dense vector
    ////////////////////////////


    // Serial SpMV //
    float *host_res = new float[m];

    clock_t begin = clock();    
    simple_spmv(host_res, vect, values, col_idx, row_off, nnz, m, n);
    clock_t end = clock();
    double cpu_time = double(end - begin) / CLOCKS_PER_SEC;
    cpu_time = cpu_time * 1000;
    ////////////////////////////


    // Device Memory allocation //
    float *d_values, *d_res, *d_vect;
    int *d_row_off, *d_col_idx; 
    cudaMalloc((void**)&d_values, sizeof(float)*nnz);
    cudaMalloc((void**)&d_col_idx, sizeof(int)*nnz);
    cudaMalloc((void**)&d_row_off, sizeof(int) * (m+1));
    cudaMalloc((void**)&d_res, sizeof(float) * m);
    cudaMalloc((void**)&d_vect, sizeof(float) * n);
    ////////////////////////////


    // Host to device copy //
    cudaMemcpy(d_values,values,sizeof(float) * nnz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx,col_idx,sizeof(int) * nnz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_off,row_off,sizeof(int) * (m+1),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vect,vect,sizeof(float) * n,cudaMemcpyHostToDevice);
    ////////////////////////////


    // Parallel SpMV //
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid((m-1)/BLOCK_SIZE + 1,1,1);

    cudaEventRecord(start);
    parallel_spmv_1<<<dimGrid,dimBlock>>> (d_values, d_col_idx, d_row_off, d_vect, d_res, m, n, nnz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    ////////////////////////////


    // Copy result to host //
    float * result_from_device = new float[m];
    cudaMemcpy(result_from_device, d_res, sizeof(float)*n,cudaMemcpyDeviceToHost);
    ////////////////////////////

    // Check Result //
    checker(result_from_device, host_res, m);
    ////////////////////////////

    // Free Device Memory //
    cudaFree(d_values);
    cudaFree(d_col_idx);
    cudaFree(d_row_off);
    cudaFree(d_res);
    cudaFree(d_vect);
    ////////////////////////////

    // Print Statistics //
    cout<<"\n\nCPU Execution time = "<<cpu_time;
    cout<<"\n\nGPU Execution time = "<<gpu_time;
    cout<<"\n\n";
    ////////////////////////////
}