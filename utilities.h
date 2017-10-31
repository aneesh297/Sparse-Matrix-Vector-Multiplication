#include <bits/stdc++.h>

using namespace std;

// Generates a sparse matrix while keeping count of non zero elements
 float * sparse_gen(int n, int &nnz, int &nnz_row, int &nnz_max)
{
	float *arr = new float[n];
	
	nnz_row = 0;

	for (int j = 0; j < n; j++)
	{

		float prob = (rand()%10)/10.0;

		//If randomly generated no (b/w 0 and 1) is not greater than 0.7 then matrix cell value is 0.
		if(prob>=0.7)
		{
			arr[j] = rand()%100 + 1;
			nnz_row++;
		}
		else
		{
			arr[j] = 0;
		}
	}

	if(nnz_row > nnz_max)
	{
		nnz_max = nnz_row;
	}

	nnz += nnz_row;

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


//Performs simple sparse matrix vector multiplication using CSR matrix
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

//Groups rows into bins. Row of size n is put in bin i where 2^(i-1)< n <= 2^i
void calculate_bin_size(vector <vector<int> > &bins, int *nnz_row, int m)
{
	for(int i = 0; i < m; i++)
	{
		int x = nnz_row[i], cnt = 0;

		while(true)
		{
			if(pow(2,cnt) >= x)
			{
				break;
			}
			cnt++;
		}
		if(x == 1)
			cnt = 1;

		bins[cnt].push_back(i);
		cout<<nnz_row[i]<<" "<<i<<endl;
	}
}