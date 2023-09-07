#include <iostream>
#include <vector>
#include <cmath>
#include "DataProc.h"
#include "omp.h"
#include <iomanip>
using namespace std;

void findCSRdiag(int* rows, int* cols, double* elem, int* diag, long nnz)
{
    for (int i = 0; i < nnz; ++i)
    {
        if (rows[i] == cols[i])
        {
            diag[rows[i]] = i;
        }
    }
}
double ILU(double* elem, int* jptr, int* iptr, int* diag, int N)
{
    double beg = omp_get_wtime();
    for (int i=1; i<N;++i)
    {
        //cout << "Row " << i << endl;
        int s1=iptr[i];
        bool pr1=true;
        vector<bool> tempvec(N);
        vector<int> tempjptr(N);
        for (int i=0;i<N;++i)
        {
            tempvec.push_back(0);
            tempjptr.push_back(0);
        }
        for (int j = s1; j<iptr[i + 1]; ++j)
        {
            tempvec[jptr[j]]=true;
            tempjptr[jptr[j]]=j;
        }
        while(pr1==true)
        {
            int k=jptr[s1];
            if (k>=i)break;
            else
            {
                elem[s1]=elem[s1]/elem[diag[k]];
                int s2=s1;
                ++s1;
                 
                int y1=s1;
                int y_end1 = iptr[i + 1];
                int y2=diag[k]+1;
                int y_end2 = iptr[k + 1];
                if (y_end1>=y1||y_end2>=y2)
                {
                    for (int j=y2;j<y_end2;++j)
                    {
                         if (tempvec [jptr [j]])
                         elem[tempjptr [jptr [j]]]-=elem [s2]*elem [j];
                    }
                }
            }
         }
    }
    double end = omp_get_wtime();
    return end - beg;
}
void print(double* elem, int* jptr, int* iptr, int N)
{
    vector<vector<double>> A;
     for(int i=0;i<N;++i)
     {
          for(int j=0;j<N;++j)
          {
              A.push_back(vector<double> (N, 0));
          }
     }
    
    for(int i=0;i<N;++i)
    {
        for(int k=iptr[i];k<iptr[i+1];++k) 
        {
            A[i][jptr [k]]=elem[k];
        }
    }
    for(int i=0;i<N;++i)
    {
        cout<<endl;
        for(int j=0;j<N;++j)
        {
            if (A[i][j] == 0.0)cout << 0 << "  ";
            else
            cout<<setprecision(2)<<A[i][j]<<" ";
        }
    }
}

void check(vector<double> &elem,vector<double> &elem1,vector<int> &jptr, vector<int> &iptr, vector<int> &diag, int N)
{
    vector<vector<double>> L, U, R;
    for(int i=0;i<N;++i)
     {
          for(int j=0;j<N;++j)
          {
              L.push_back(vector<double> (N, 0));
              U.push_back(vector<double> (N, 0));
              R.push_back(vector<double> (N, 0));
          }
     }
    for(int i=0;i<N;++i)
    {
        for(int k=iptr[i];k<iptr[i+1];++k) 
        {
            if (k==diag[i])
            {
                 U[i][jptr [k]]=elem[k];
                 L[i][jptr [k]]=1;
            }
            if (k>diag[i]) U[i][jptr [k]]=elem[k];
            if (k<diag[i]) L[i][jptr [k]]=elem[k];
        }
    }
    cout<<"\nL:";
    for(int i=0;i<N;++i)
    {
        cout<<endl;
        for(int j=0;j<N;++j)
            cout<<L[i][j]<<" ";
    }
    cout<<"\nU:";
    for(int i=0;i<N;++i)
    {
        cout<<endl;
        for(int j=0;j<N;++j)
            cout<<U[i][j]<<" ";
    }
   //умножение
   for (int i=0;i<N;++i)
       for (int j=0;j<N;++j)
           for (int k=0;k<N;++k)
               R[i][j]+=L[i][k]*U[k][j];
    cout<<"\nR:";
    for(int i=0;i<N;++i)
    {
        cout<<endl;
        for(int j=0;j<N;++j)
            cout<<R[i][j]<<" ";
    }
    cout<<"\nnorm:\n";
    double norm=0;
    for (int i=0;i<N;++i)
    {
        norm+=(elem1[i]-elem[i])*(elem1[i]-elem[i]);
    }
    cout<<sqrt(norm);
}

int main(int argc, char *argv[]) 
{
	
    string file_path_mat = "mat.mtx";
    int base = 0;
    long num_rows, num_cols, nnz, num_lines;
    int is_symmetric;
    double time_beg, time_data_parsing, time_IC_beg, time_IC_end, time_CG, time_end;
    time_beg = omp_get_wtime();

    mtx_header(/*argv[1]*/file_path_mat, &num_lines, &num_rows, &num_cols, &nnz, &is_symmetric);

    cout << "\nmatrix name: " << file_path_mat << endl <<
        "num. rows: " << num_rows << endl <<
        "num. cols: " << num_cols << endl <<
        "nnz: " << nnz << endl <<
        "structure: " << ((is_symmetric) ? "symmetric" : "unsymmetric") << endl;
    /* argv[1] */
    if (num_rows != num_cols)
    {
        cout << "the input matrix must be square\n";
        return EXIT_FAILURE;
    }
    if (!is_symmetric)
    {
        cout << "the input matrix must be symmetric\n";
        return EXIT_FAILURE;
    }

    int     m = num_rows;
    int     num_offsets = m + 1;
    int* h_A_rows = new int[nnz];
    int* h_A_columns = new int[nnz];
    double* h_A_values = new double[nnz];
    double* h_X = new double[m];
    double* h_b = new double[m];
    int* h_diag = new int[m];

    printf("Matrix parsing...\n");

    mtx_parsing(/*argv[1]*/file_path_mat, num_lines, num_rows, nnz, h_A_rows,
        h_A_columns, h_A_values, base);
    cout << "Done" << endl;
    //cout << "\nunsorted coo:\n";
    
    cudaSortCooByRow(nnz, m, m, h_A_rows, h_A_columns, h_A_values);
    //cout << "\nsorted coo:\n";
   /*for (int i = 0; i < nnz; ++i)
    {
        cout << h_A_rows[i] << " " << h_A_columns[i] << " " << h_A_values[i] << endl;
    }*/ 
    findCSRdiag(h_A_rows, h_A_columns, h_A_values, h_diag, nnz);
   /*cout << "diag: " << endl;
    for (int i = 0; i < 20; ++i)
    {
        cout << h_diag[i] << endl;
    }*/

    int* iptr = new int[num_rows + 1];
    cudaCoo2csr(h_A_rows, nnz, m, iptr);
    print(h_A_values, h_A_columns, iptr, m);
   // time_data_parsing = omp_get_wtime();
    //cout << "ILU:" << endl;
	cout<<"\nTotal time: "<<ILU(h_A_values, h_A_columns, iptr, h_diag, m)<<" c";
    print(h_A_values, h_A_columns, iptr, m);
}