#include "io.h"
#include "utilities.h"
#include <math.h>

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
    
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;

    int row = warp_id;

    if(row < m){
        int begin_index = row_off[row];
        int end_index = row_off[row+1];

        float thread_sum = 0.0;
        for(int i = begin_index + lane_id; i < end_index; i+=32)
            thread_sum += values[i] * vect[col_idx[i]];
        
        thread_sum += __shfl_down(thread_sum,16);
        thread_sum += __shfl_down(thread_sum,8);
        thread_sum += __shfl_down(thread_sum,4);
        thread_sum += __shfl_down(thread_sum,2);
        thread_sum += __shfl_down(thread_sum,1);

        if(lane_id == 0)
            res[row] = thread_sum;
        
    }
}
////////////////////////////

// Parallel SpMV with Average threads per row
__global__
void parallel_spmv_3(float * values, int * col_idx, int * row_off, float * vect, float * res, 
    int m, int n, int nnz, int threads_per_row){
    
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int vector_id = thread_id / threads_per_row;
    int lane_id = thread_id % threads_per_row;

    int row = vector_id;

    if(row < m){
        int begin_index = row_off[row];
        int end_index = row_off[row+1];

        float thread_sum = 0.0;
        for(int i = begin_index + lane_id; i < end_index; i+=threads_per_row)
            thread_sum += values[i] * vect[col_idx[i]];

        int temp = threads_per_row/2;
        while(temp >= 1){
            thread_sum += __shfl_down(thread_sum, temp);
            temp/=2;
        }

        if(lane_id == 0)
            res[row] = thread_sum;

    }
}
////////////////////////////


// Utility function to calculate thread_per_row for parallel_spmv_3 //
int nearest_pow_2(float n){
    int lg = (int)log2(n);
    return (int)pow(2,lg);
}
////////////////////////////


int main(){

    // Create Cuda Events //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    ////////////////////////////

    // Reading Dataset //
    int m,n,nnz,nnz_max,nnz_avg;

    conv(nnz,m,n,nnz_max,nnz_avg);  // Defined in io.h

    cout<<"\nrows    = "<<m;
    cout<<"\ncolumns = "<<n;
    cout<<"\nnnz     = "<<nnz;
    cout<<"\nnnz_max = "<<nnz_max;
    cout<<"\nnnz_avg = "<<nnz_avg;
    cout<<"\n\n";

    float *vect = vect_gen(n); //generating dense vector
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
    ////////////////////////////
    dim3 dimBlock(BLOCK_SIZE,1,1);
    dim3 dimGrid_1((m-1)/BLOCK_SIZE + 1,1,1);
    dim3 dimGrid_2((m-1)/32 + 1,1,1);
    int threads_per_row = min(32, nearest_pow_2(nnz_avg));
    dim3 dimGrid_3((m-1)/(1024/threads_per_row)+1, 1, 1);

    // Calling one thread per row kernel
    cudaEventRecord(start);
    parallel_spmv_1<<<dimGrid_1,dimBlock>>> (d_values, d_col_idx, d_row_off, d_vect, d_res, m, n, nnz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_1 = 0;
    cudaEventElapsedTime(&gpu_time_1, start, stop);

    // calling one warp per row kernel
    cudaEventRecord(start);
    parallel_spmv_2<<<dimGrid_2,dimBlock>>> (d_values, d_col_idx, d_row_off, d_vect, d_res, m, n, nnz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_2 = 0;
    cudaEventElapsedTime(&gpu_time_2, start, stop);

    // calling avg threads per row
    cudaEventRecord(start);
    parallel_spmv_3<<<dimGrid_3,dimBlock>>> (d_values, d_col_idx, d_row_off, d_vect, d_res, m, n, nnz, threads_per_row);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_3 = 0;
    cudaEventElapsedTime(&gpu_time_3, start, stop);

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
    cout<<"\n\nCPU Execution time                  = "<<cpu_time<<" ms";
    cout<<"\n\nGPU Execution time - Thread per Row = "<<gpu_time_1<<" ms";
    cout<<"\n\nGPU Execution time - Warp per Row   = "<<gpu_time_2<<" ms";
    cout<<"\n\n\nThreads per row in avrg per row     = "<<threads_per_row;
    cout<<"\nGPU Execution time - Avrg per Row   = "<<gpu_time_3<<" ms";
    cout<<"\n\n";
    ////////////////////////////
}