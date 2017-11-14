# Sparse-Matrix-Vector-Multiplication

## Contributors : 
* Aneesh Aithal
* Sagar Bharadwaj


## Explanation
Sparse matrix-vector multiplication (SpMV) is a widely used computational kernel. The most commonly used format for a sparse matrix is CSR (Compressed Sparse Row), but a number of other representations have recently been developed that achieve higher SpMV performance. However, the alternative representations typically impose a significant preprocessing over-head. While a high preprocessing overhead can be amortized for applications requiring many iterative invocations of SpMV that use the same matrix, it is not always feasible – for instance when analyzing large dynamically evolving graphs.
<br/>
Ashari et al proposed an algorithm named *ACSR* that uses the standard CSR format for Sparse-Matrix multiplication. In *ACSR*, thread divergence is reduced by grouping rows with similar number of zeros into a bin. It also uses dynamic parallelism for rows that span a wide range of non zero counts.
<br/>
SpMV is also implemented using cuSparse library for comparision with ACSR.
<br/>

The recorded values are : 

**Recorded values table**

To compile use:
`nvcc mainfil.cu -arch=sm_50 -rdc=true`

Datasets can be found here: https://sparse.tamu.edu/
