#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cusparse_v2.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}



int cudaSortCooByRow(int nnz,int num_rows, int num_columns,int* rows,int* cols,double* vals)
{
 
    int* s_permutation = new int[nnz];

    //--------------------------------------------------------------------------
    // Device memory management
    int*    d_rows=0,   * d_columns=0,     * d_permutation=0;
    double* d_values=0, * d_values_sorted=0;
    void*   d_buffer=0;
    size_t  bufferSize;
    CHECK_CUDA(cudaMalloc((void**)&d_rows,          nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_columns,       nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_values,        nnz * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_values_sorted, nnz * sizeof(double)))
    CHECK_CUDA(cudaMalloc((void**)&d_permutation,   nnz * sizeof(int)))

    CHECK_CUDA(cudaMemcpy(d_rows,    rows, nnz * sizeof(int),
        cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_columns, cols, nnz * sizeof(int),
        cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_values,  vals, nnz * sizeof(double),
        cudaMemcpyHostToDevice))
    //--------------------------------------------------------------------------
    // CUSPARSE APIs solving part
    cusparseHandle_t handle = 0;
    cusparseSpVecDescr_t vec_permutation=0;
    cusparseDnVecDescr_t vec_values=0;
    cusparseCreate(&handle);

    // Create sparse vector for the permutation
    CHECK_CUSPARSE(cusparseCreateSpVec(&vec_permutation, nnz, nnz,
        d_permutation, d_values_sorted,CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
    // Create dense vector for wrapping the original coo values
    CHECK_CUSPARSE(cusparseCreateDnVec(&vec_values, nnz, d_values,CUDA_R_64F))

    // Query working space of COO sort
    CHECK_CUSPARSE(cusparseXcoosort_bufferSizeExt(handle, num_rows, num_columns, nnz, d_rows,
            d_columns, &bufferSize))
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize))
    // Setup permutation vector to identity
    CHECK_CUSPARSE(cusparseCreateIdentityPermutation(handle, nnz,d_permutation))
    CHECK_CUSPARSE(cusparseXcoosortByRow(handle, num_rows, num_columns, nnz,
            d_rows, d_columns, d_permutation,d_buffer))
    CHECK_CUSPARSE(cusparseGather(handle, vec_values, vec_permutation))
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpVec(vec_permutation))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vec_values))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // copy result 
    CHECK_CUDA(cudaMemcpy(rows, d_rows,                 nnz * sizeof(int),
        cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(cols, d_columns,              nnz * sizeof(int),
        cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(vals, d_values_sorted,        nnz * sizeof(double), 
        cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(s_permutation, d_permutation, nnz * sizeof(int), 
        cudaMemcpyDeviceToHost))
        
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(d_rows))
    CHECK_CUDA(cudaFree(d_columns))
    CHECK_CUDA(cudaFree(d_permutation))
    CHECK_CUDA(cudaFree(d_values))
    CHECK_CUDA(cudaFree(d_values_sorted))
    CHECK_CUDA(cudaFree(d_buffer))
    
    delete[]s_permutation;
        return EXIT_SUCCESS;
}

int cudaCoo2csr(int* cooRowIdx,int nnz,int m,int* iptr)
{
    cusparseHandle_t handle = 0;

    int* d_iptr = 0;
    int* d_rows = 0;
    //CUSPARSE API init
    CHECK_CUDA(cudaMalloc((void**)&d_iptr, (static_cast<unsigned long long>(m) + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**)&d_rows, nnz * sizeof(int)))
    CHECK_CUDA(cudaMemcpy(d_rows, cooRowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUSPARSE(cusparseCreate(&handle))
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, d_rows, nnz, m, d_iptr, CUSPARSE_INDEX_BASE_ZERO))
    CHECK_CUDA(cudaMemcpy(iptr, d_iptr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost))
    //Free resourses
    CHECK_CUSPARSE(cusparseDestroy(handle))
    CHECK_CUDA(cudaFree(d_iptr))
    CHECK_CUDA(cudaFree(d_rows))
        return EXIT_SUCCESS;
}

void mtx_header(
    string file_path,
    long* num_lines,
    long* num_rows,
    long* num_cols,
    long* nnz,
    int* is_symmetric) {
    string buffer;
    ifstream ifile(file_path, ios::in);
    if (ifile.is_open())
    {
        getline(ifile, buffer);
        stringstream ss(buffer);
        string s;
        ss >> s;
        if (s != "%%MatrixMarket")
        {
            cout << "Unsupported file format. Only MTX format is supported";
            exit(EXIT_FAILURE);
        }
        ss >> s >> s >> s;
        if (s != "real")
        {
            cout << "Only real (double) matrices are supported";
            exit(EXIT_FAILURE);
        }
        ss >> s;
        *is_symmetric = (s == "symmetric" ? 1 : 0);
        char c;
        ifile.get(c);
        while (c == '%')
        {
            getline(ifile, buffer);
            ifile.get(c);
        }
        getline(ifile, buffer);
        buffer.insert(0, 1, c);
        stringstream ss1(buffer);
        ss1 >> *num_rows >> *num_cols >> *num_lines;
        *nnz = (*is_symmetric) ? *num_lines * 2 - *num_rows : *num_lines;
    }
    else
    {
        cout << "Error: unable to open the file " << file_path;
        exit(EXIT_FAILURE);
    }
}
void mtx_vec_parsing(
    string file_path,
    int         num_lines,
    double* vec)
{
    string buffer;
    ifstream ifile(file_path, ios::in);
    if (ifile.is_open())
    {
        getline(ifile, buffer);
        stringstream ss(buffer);
        string s;
        ss >> s;
        if (s != "%%MatrixMarket")
        {
            cout << "Unsupported file format. Only MTX format is supported";
            exit(EXIT_FAILURE);
        }
        ss >> s >> s >> s;
        if (s != "real")
        {
            cout << "Only real (double) matrices are supported";
            exit(EXIT_FAILURE);
        }
        char c;
        ifile.get(c);
        while (c == '%')
        {
            getline(ifile, buffer);
            ifile.get(c);
        }
        getline(ifile, buffer);
        buffer.clear();

        for (int i = 0; i < num_lines; ++i)
        {
            getline(ifile, buffer);
            stringstream ss(buffer);
            ss >> vec[i];
        }
    }
    else
    {
        cout << "Error: unable to open the file " << file_path;
        exit(EXIT_FAILURE);
    }
}

void mtx_parsing(
    string file_path,
    int         num_lines,
    int         num_rows,
    int         nnz,
    int* iptr,
    int* jptr,
    double* values,
    int         base)
{
    string buffer;
    ifstream ifile(file_path, ios::in);
    if (ifile.is_open())
    {
        char c;
        ifile.get(c);
        while (c == '%')
        {
            getline(ifile, buffer);
            ifile.get(c);
        }
        getline(ifile, buffer);
        buffer.clear();

        for (int i = 0, up_count = 0; i < num_lines; ++i)
        {
            getline(ifile, buffer);
            stringstream ss(buffer);
            int    row, column;
            double value;
            ss >> row;
            ss.ignore(1);
            ss >> column;
            ss.ignore(1);
            ss >> value;
            row -= (1 - base);
            column -= (1 - base);

            iptr[i] = row;
            jptr[i] = column;
            values[i] = value;
            if (nnz != num_lines && column != row) { // is stored symmetric

                iptr[up_count + num_lines] = column;
                jptr[up_count + num_lines] = row;
                values[up_count + num_lines] = value;
                up_count++;
            }
        }
    }
    else
    {
        cout << "Error: unable to open the file " << file_path;
        exit(EXIT_FAILURE);
    }
}