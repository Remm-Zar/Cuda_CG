
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
//#include <stdlib.h> // EXIT_FAILURE
//#include <string.h> // strtok
#include <sstream>
#include <fstream>
#include "device_launch_parameters.h"
#include <iostream>
#include "DataProc.h"
#include "CG_solver.h"
#include <vector>
#include <iomanip>
#include "omp.h"
using namespace std;

void print(double* elem, int* jptr, int* iptr, int N)
{
    vector<vector<double>> A;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A.push_back(vector<double>(N, 0));
        }
    }

    for (int i = 0; i < N; ++i)
    {
        for (int k = iptr[i]; k < iptr[i + 1]; ++k)
        {
            A[i][jptr[k]] = elem[k];
        }
    }
    for (int i = 0; i < N; ++i)
    {
        cout << endl;
        for (int j = 0; j < N; ++j)
        {
            if (A[i][j] == 0.0)cout << 0 << "       ";
            else
                cout << setprecision(2) << A[i][j] << " ";
        }
    }
}

int main(int argc, char** argv) {
    const int    maxIterations = 2000000;
    const double tolerance = 1e-8f;
    //it's possible to use args from command line, but not now
   /* printf("Usage: cg_example <matrix.mtx>\n");
    if (argc != 2) {
        printf("Wrong parameter: cg_example <matrix.mtx>\n");
        return EXIT_FAILURE;
    }*/
    string file_path_mat = "matrixData/parabolic_fem.mtx";
    string file_path_vec = "matrixData/parabolic_fem_b.mtx";
    int base = 0;
    long num_rows, num_cols, nnz, num_lines;
    int is_symmetric;
    double time_beg,time_data_parsing,time_IC_beg,time_IC_end,time_CG,time_end;
    time_beg = omp_get_wtime();

    mtx_header(/*argv[1]*/file_path_mat, &num_lines, &num_rows, &num_cols, &nnz, &is_symmetric);

    cout << "\nmatrix name: " << file_path_mat << endl <<
        "num. rows: " << num_rows << endl <<
        "num. cols: " << num_cols << endl <<
        "nnz: " << nnz << endl <<
        "structure: " << ((is_symmetric) ? "symmetric" : "unsymmetric")<<endl;
        /* argv[1] */
    if (num_rows != num_cols) 
    {
        cout<<"the input matrix must be square\n";
        return EXIT_FAILURE;
    }
    if (!is_symmetric) 
    {
        cout<<"the input matrix must be symmetric\n";
        return EXIT_FAILURE;
    }

    int     m = num_rows;
    int     num_offsets = m + 1;
    int*    h_A_rows = new int[nnz];
    int*    h_A_columns = new int[nnz];
    double* h_A_values = new double[nnz];
    double* h_X = new double[m];
    double* h_b = new double[m];

    cout << "Vector parsing..." << endl;
    mtx_vec_parsing(file_path_vec, m, h_b);
    cout << "Done" << endl;
    for (int i = 0; i < 20; ++i)
    {
        cout << h_b[i] << endl;
    }
    printf("Matrix parsing...\n");
    mtx_parsing(/*argv[1]*/file_path_mat, num_lines, num_rows, nnz, h_A_rows,
        h_A_columns, h_A_values, base); 
    cout << "Done" << endl;
    ////DEBUG
   /*cout << "\nunsorted coo:\n";
    for (int i = 0; i < 20; ++i)
    {
        cout << h_A_rows[i] << " " << h_A_columns[i] << " " << h_A_values[i] << endl;
    }*/ 
    ////
    cudaSortCooByRow(nnz, m, m, h_A_rows, h_A_columns, h_A_values);
    cout << "\nsorted coo:\n";
    for (int i = 0; i < 20; ++i)
    {
        cout << h_A_rows[i] << " " << h_A_columns[i] << " " << h_A_values[i] << endl;
    }
   
    int* iptr = new int[num_rows + 1];
    cudaCoo2csr(h_A_rows, nnz, m, iptr);
    time_data_parsing= omp_get_wtime();
    printf("Testing CG\n");
    
    //for (int i = 0; i < num_rows; i++)
    //     h_X[i] = 1.0;
    //--------------------------------------------------------------------------
    // ### Device memory management ###
    int* d_A_rows, * d_A_columns;
    double* d_A_values, * d_L_values;
    Vec     d_B, d_X, d_R, d_R_aux, d_P, d_T, d_tmp;

    // allocate device memory for CSR matrices
    CHECK_CUDA(cudaMalloc((void**)&d_A_rows,    num_offsets * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_A_columns, nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_A_values,  nnz * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_L_values,  nnz * sizeof(double)))

    CHECK_CUDA(cudaMalloc((void**)&d_B.ptr,     m * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_X.ptr,     m * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_R.ptr,     m * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_R_aux.ptr, m * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_P.ptr,     m * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_T.ptr,     m * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_tmp.ptr,   m * sizeof(double)))

    // copy the CSR matrices and vectors into device memory
    CHECK_CUDA(cudaMemcpy(d_A_rows,    iptr,        num_offsets * sizeof(int),
        cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_A_columns, h_A_columns, nnz * sizeof(int),
        cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_A_values,  h_A_values,  nnz * sizeof(double),
        cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_L_values,  h_A_values,  nnz * sizeof(double),
        cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_X.ptr,     h_X,         m * sizeof(double),
        cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_B.ptr,     h_b,         m*sizeof(double),
        cudaMemcpyHostToDevice))

        
    //--------------------------------------------------------------------------
    // ### cuSPARSE Handle and descriptors initialization ###
    // create the test matrix on the host
    cublasHandle_t   cublasHandle = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUBLAS(cublasCreate(&cublasHandle))
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle))
    // Create dense vectors
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_B.vec,     m, d_B.ptr, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_X.vec,     m, d_X.ptr, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_R.vec,     m, d_R.ptr, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_R_aux.vec, m, d_R_aux.ptr,CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_P.vec,     m, d_P.ptr, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_T.vec,     m, d_T.ptr, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseCreateDnVec(&d_tmp.vec,   m, d_tmp.ptr, CUDA_R_64F))

    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    cusparseSpMatDescr_t matA, matL;
    int* d_L_rows = d_A_rows;
    int* d_L_columns = d_A_columns;
    cusparseFillMode_t   fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    // A
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, m, m, nnz, d_A_rows,
        d_A_columns, d_A_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        baseIdx, CUDA_R_64F))
    // L
    CHECK_CUSPARSE(cusparseCreateCsr(&matL, m, m, nnz, d_L_rows,
            d_L_columns, d_L_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            baseIdx, CUDA_R_64F))
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE,
            &fill_lower, sizeof(fill_lower)))
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE,
            &diag_non_unit, sizeof(diag_non_unit)))
    //--------------------------------------------------------------------------
    // ### Preparation ### 
   void* d_bufferMV;
    const double alpha = 0.95;
    size_t       bufferSizeMV;
    
    double       beta = 0.0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
    cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
    CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV))
    CHECK_CUDA(cudaMalloc(&d_bufferMV, bufferSizeMV))

    CHECK_CUSPARSE(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))

    // X0 = 0
    CHECK_CUDA(cudaMemset(d_X.ptr, 0x0, m * sizeof(double)))
    time_IC_beg = omp_get_wtime();
    //--------------------------------------------------------------------------
    // IC factorization of A (csric0) -> L, L^T
    cusparseMatDescr_t descrM;
    csric02Info_t      infoM = NULL;
    int                bufferSizeIC = 0;
    void* d_bufferIC;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrM))
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrM, baseIdx))
    CHECK_CUSPARSE(cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL))
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER))
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrM,CUSPARSE_DIAG_TYPE_NON_UNIT))
    CHECK_CUSPARSE(cusparseCreateCsric02Info(&infoM))

    CHECK_CUSPARSE(cusparseDcsric02_bufferSize(cusparseHandle, m, nnz, descrM, d_L_values,
        d_A_rows, d_A_columns, infoM, &bufferSizeIC))
    CHECK_CUDA(cudaMalloc(&d_bufferIC, bufferSizeIC))
    CHECK_CUSPARSE(cusparseDcsric02_analysis(cusparseHandle, m, nnz, descrM, d_L_values,
        d_A_rows, d_A_columns, infoM,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC))
    int structural_zero;
    CHECK_CUSPARSE(cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
        &structural_zero))
        cout << "structural_zero: " << structural_zero << endl;
       
    // M = L * L^T
    CHECK_CUSPARSE(cusparseDcsric02(
        cusparseHandle, m, nnz, descrM, d_L_values,
        d_A_rows, d_A_columns, infoM,
        CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC))
        double* h_L = new double[nnz];
    ///DEBUG
        /*cudaMemcpy(h_L, d_L_values, nnz * sizeof(double), cudaMemcpyDeviceToHost);
        cout << "\nL:\n";
        for (int i = 0; i < 20; ++i)
        {
            cout << h_L[i] << endl;
        }
        delete[]h_L;
        int numerical_zero;*/

    //CHECK_CUSPARSE(cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
    //    &numerical_zero))
     //   cout << "numerical_zero" << numerical_zero << endl;;

    CHECK_CUSPARSE(cusparseDestroyCsric02Info(infoM))
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrM))
    CHECK_CUDA(cudaFree(d_bufferIC)) 
    // Find numerical zero
    
    time_IC_end = omp_get_wtime();
    //--------------------------------------------------------------------------
    // ### Run CG computation ###
    printf("CG solving:\n");
    gpu_CG(cublasHandle, cusparseHandle, m,
    matA, matL, d_B, d_X, d_R, d_R_aux, d_P, d_T,
    d_tmp, maxIterations, tolerance);
    time_CG = omp_get_wtime();
//--------------------------------------------------------------------------
// ### Free resources ###
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_B.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_X.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_R.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_R_aux.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_P.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_T.vec))
    CHECK_CUSPARSE(cusparseDestroyDnVec(d_tmp.vec))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroySpMat(matL))
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle))
    CHECK_CUBLAS(cublasDestroy(cublasHandle))

    delete[](iptr);
    delete[](h_A_columns);
    delete[](h_A_rows);
    delete[](h_A_values);
    delete[](h_X);

    CHECK_CUDA(cudaFree(d_X.ptr))
    CHECK_CUDA(cudaFree(d_B.ptr))
    CHECK_CUDA(cudaFree(d_R.ptr))
    CHECK_CUDA(cudaFree(d_R_aux.ptr))
    CHECK_CUDA(cudaFree(d_P.ptr))
    CHECK_CUDA(cudaFree(d_T.ptr))
    CHECK_CUDA(cudaFree(d_tmp.ptr))
    CHECK_CUDA(cudaFree(d_A_values))
    CHECK_CUDA(cudaFree(d_A_columns))
    CHECK_CUDA(cudaFree(d_A_rows))
    CHECK_CUDA(cudaFree(d_L_values))
    
    time_end= omp_get_wtime();
    cout << "\nData processing: " << time_data_parsing - time_beg << " c" << endl;
    cout << "IC factorization: " << time_IC_end - time_IC_beg << " c" << endl;
    cout << "CG: " << time_CG - time_IC_end << " c" << endl;
    cout << "Total: " << time_end - time_beg << " c" << endl;
        return EXIT_SUCCESS;
}