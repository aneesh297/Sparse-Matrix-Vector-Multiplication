#include <bits/stdc++.h>

using namespace std;

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
float *vect_gen(int n, bool isSerial = false)
{
 float *vect = new float[n];

 for(int i = 0; i < n; i++)
 {
   if(!isSerial)
    vect[i] = rand()%10;
  else
    vect[i] = i+1;
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
void to_csr(vector<vector<float> > mat, float *values, int *col_idx, int *row_off, int m, int n)
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
int checker(float *arr1, float *arr2, int size)
{
	float err = 0;
	int count = 0;

	for(int i = 0; i < size; i++)
	{
		err = abs(arr1[i] - arr2[i]);
		if(err > 0.1){
      if(count<10)
      {
        cout<<"Incorrect ";
        cout<<arr1[i]<<" "<<arr2[i]<<" "<<err<<" "<<i<<endl;
      }
			
			 count++;
		}
	}

  if(count == 0)
	cout<<"Correct Result\n\n";
  else
    cout<<"no of Incorrect = "<<count<<endl;
	return 0;
}

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
