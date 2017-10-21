#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

// Generates a sparse matrix while keeping count of non zero elements
 float * sparse_gen(int n, int &nnz)
{
	float *arr = new float[n];
	

	for (int j = 0; j < n; j++)
	{

		float prob = (rand()%10)/10.0;

		if(prob>=0.7)
		{
			arr[j] = rand()%100 + 1;
			nnz++;
		}
		else
		{
			arr[j] = 0;
		}
	}

	return arr;
}

//Generates the dense vector required for multiplication
float *vect_gen(int n)
{
	float *vect = new float[n];

	for(int i = 0; i < n; i++)
	{
		vect[i] = rand()%100;
	}

	return vect;
}

//Prints a matrix
void display_matrix(float **mat, int m, int n)
{
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			cout<<mat[i][j]<<" ";
		}
		cout<<endl;
	}
}


//Prints any vector
void display_vector(float *vect, int n)
{
	for(int i = 0; i < n; i++)
	{
		cout<<vect[i]<<" ";
	}

	cout<<endl;	
}

//Converts a matrix to the Compressed Sparse Row format
void to_csr(float *mat[], float *values, int *col_idx, int *row_off, int m, int n)
{
	int cnt = 0;
	for(int i = 0; i < m; i++)
	{
		int flag = 1;
		for(int j = 0; j < n; j++)
		{
			if(mat[i][j]!=0)
			{
				values[cnt] = mat[i][j];
				col_idx[cnt] = j;
				cnt++;

				if(flag)
				{
					flag = 0;
					row_off[i] = cnt-1;
				}
			}
		}
		if(flag)
			row_off[i] = cnt;
	}
}

//Displays the 3 CSR arrays
void display_csr(float * values, int *col_idx, int *row_off,int nnz,  int m)
{
	cout<<"\nValues: \n";
	for(int i = 0; i < nnz; i++)
	{
		cout<<values[i]<<" ";
	}
	cout<<"\nColumn Indices: \n";
	for(int i = 0; i < nnz; i++)
	{
		cout<<col_idx[i]<<" ";
	}
	cout<<"\nRow Offsets: \n";
	for(int i = 0; i < m; i++)
	{
		cout<<row_off[i]<<" ";
	}
	cout<<endl;
}


//Performs simple sparse matrix vector multiplication
void simple_spmv(float *res, float *vect, float * values, int *col_idx, int *row_off, int nnz, int m,int n)
{
	for(int i = 0; i < n; i++)
	{
		res[i] = 0;
		if(row_off[i] != nnz && row_off[i] != row_off[i+1])
		{
			int cnt = 0;
			if(i!=n-1)
			{
				cnt = row_off[i+1] - row_off[i];
			}
				else
				{
					cnt = nnz - row_off[i];
				}

			for(int j = 0; j < cnt; j++)
			{
				res[i] += values[row_off[i] + j] * vect[col_idx[row_off[i] + j]];
			}
		}
	}
}

int main()
{
	srand (time(NULL));
	int m = 5, n = 5;
	int nnz = 0; 
	float *mat[m], *vect, *res;
	float *values;
	int *col_idx, *row_off;

	for(int i = 0; i < m; i++)
	{
		mat[i] = sparse_gen(n, nnz);
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



	to_csr(mat, values, col_idx, row_off, m, n);
	display_csr(values, col_idx, row_off, nnz, m);


	res = new float[n];

	simple_spmv(res, vect, values, col_idx, row_off, nnz, m, n);

	cout<<"Result vector: \n";
	display_vector(res, n);



}