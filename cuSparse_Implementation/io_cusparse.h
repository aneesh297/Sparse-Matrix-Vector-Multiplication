#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

int *col_idx, *row_off;
double *values;

void sort(int *col_idx, double *a, int start, int end)
{
  int i, j, it;
  int dt;

  for (i=end-1; i>start; i--)
    for(j=start; j<i; j++)
      if (col_idx[j] > col_idx[j+1]){

	if (a){
	  dt=a[j];
	  a[j]=a[j+1];
	  a[j+1]=dt;
        }
	it=col_idx[j];
	col_idx[j]=col_idx[j+1];
	col_idx[j+1]=it;

      }
}

void coo2csr(int row_length, int nnz, double *values, int *row, int *col,
	     double *csr_values, int *col_idx, int *row_start)
{
  int i, l;

  for (i=0; i<=row_length; i++) row_start[i] = 0;

  /* determine row lengths */
  for (i=0; i<nnz; i++) row_start[row[i]+1]++;


  for (i=0; i<row_length; i++) row_start[i+1] += row_start[i];


  /* go through the structure  once more. Fill in output matrix. */
  for (l=0; l<nnz; l++){
    i = row_start[row[l]];
    csr_values[i] = values[l];
    col_idx[i] = col[l];
    row_start[row[l]]++;
  }

  /* shift back row_start */
  for (i=row_length; i>0; i--) row_start[i] = row_start[i-1];

  row_start[0] = 0;

  for (i=0; i<row_length; i++){
    sort (col_idx, csr_values, row_start[i], row_start[i+1]);
  }

}

static void conv(int &nnz, int &row_length, int &column_length, int &nnz_max)
{

	cout<<"Enter dataset name : ";
	string d;
	cin>>d;
	d = "datasets/" + d + ".mtx";
	std::ifstream fin(d.c_str());
	if(!fin){
		cout<<"File Not found\n";
		exit(0);
	}

	//int row_length, column_length, nnz;

	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');

	// Read defining parameters:
	fin >> row_length >> column_length >> nnz;

	// Create your matrix:
	int *row, *column;
	double *coovalues;
	row = new int[nnz];
	column = new int[nnz];
	coovalues = new double[nnz];
	values = (double*) malloc(sizeof(double)*nnz);
	col_idx = (int*) malloc(sizeof(int)*nnz);
	row_off = (int*) malloc(sizeof(int)*(row_length+1));
	std::fill(row, row + nnz, 0);
	std::fill(column, column + nnz, 0);
	std::fill(values, values + nnz, 0);

	// Read the data
	for (int l = 0; l < nnz; l++)
	{
		int m, n;
		double data;
		fin >> m >> n;// >> data;

		row[l] = m;
		column[l] = n;
		coovalues[l] = rand()%10 + 1;
	}

	coo2csr(row_length, nnz, coovalues, row, column, values, col_idx, row_off);

	nnz_max = 0;

	for(int i = 0;i < -1 + row_length; i++)
	{
		if((row_off[i+1] - row_off[i]) > nnz_max)
			nnz_max = (row_off[i+1] - row_off[i]);
	}

	if((nnz - row_off[row_length-1]) > nnz_max)
		nnz_max = nnz - row_off[row_length-1];

	row_off[row_length] = nnz;

	delete []row;
	delete []column;
	delete []coovalues;

	fin.close();
}
