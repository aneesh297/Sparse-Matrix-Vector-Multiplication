# Sparse-Matrix-Vector-Multiplication

## Contributors :
* Aneesh Aithal
* Sagar Bharadwaj


## Explanation
### Brief<br/>
This repository contains an implementation of a parallel algorithm for Sparse-Matrix-Vector-Multiplication for CUDA enabled GPUs.
<br/><br/>
### Detailed explanation<br/>
Sparse matrix-vector multiplication (SpMV) is a widely used computational kernel. The most commonly used format for a sparse matrix is **CSR (Compressed Sparse Row)**, but a number of other representations have recently been developed that achieve higher SpMV performance. However, the alternative representations typically impose a significant preprocessing over-head. While a high preprocessing overhead can be amortized for applications requiring many iterative invocations of SpMV that use the same matrix, it is not always feasible – for instance when analyzing large dynamically evolving graphs.
<br/><br />
Ashari et al proposed an algorithm named **ACSR** that uses the standard CSR format for Sparse-Matrix multiplication. In *ACSR*, thread divergence is reduced by grouping rows with similar number of zeros into a bin. It also uses dynamic parallelism for rows that span a wide range of non zero counts.
<br/><br />
SpMV is also implemented using `cuSparse` library for comparision with ACSR.
<br/><br />

## Observations
The values recorde are : <br/>
**Insert recorded values table**<br/><br/>

Graphs plotted for the above recorded values can be found here : [Graphs Page](Observations/graphs.html)

## Datasets
Datasets can be found here: https://sparse.tamu.edu/
<br/>Download datasets into a directory named `datasets` in the root folder of this repository.

## Usage and Compilation
### Compliling and running ACSR
```
nvcc ACSR_Implementation/ACSR.cu -arch=sm_50
./a.out
```
(Enter the name of the dataset when prompted)

### Compiling and running cuSparse implementation of SpMV
```
nvcc cuSparse_Implementation/cuSparse_implementation.cu -lcusparse
./a.out
```
(Enter the name of the dataset when prompted)
