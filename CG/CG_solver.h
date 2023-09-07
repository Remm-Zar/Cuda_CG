#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok
#include <sstream>
#include <fstream>
#include "device_launch_parameters.h"
#include <iostream>
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
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#if defined(NDEBUG)
#   define PRINT_INFO(var)
#else
#   define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

typedef struct VecStruct {
    cusparseDnVecDescr_t vec;
    double* ptr;
} Vec;

int gpu_CG(
    cublasHandle_t       cublasHandle,
    cusparseHandle_t     cusparseHandle,
    int                  m,
    cusparseSpMatDescr_t matA,
    cusparseSpMatDescr_t matL,
    Vec                  d_B,
    Vec                  d_X,
    Vec                  d_R,
    Vec                  d_R_aux,
    Vec                  d_P,
    Vec                  d_T,
    Vec                  d_tmp,

    int                  maxIterations,
    double               tolerance)
{
    const double zero = 0.0;
    const double one = 1.0;
    const double minus_one = -1.0;
    size_t       bufferSizeMV;
    void* d_bufferMV;
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    CHECK_CUDA(cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double), cudaMemcpyDeviceToDevice))
       
        //    (b) compute R = -A * X0 + R
       CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &minus_one, matA, d_X.vec, &one, d_R.vec,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV))
        CHECK_CUDA(cudaMalloc(&d_bufferMV, bufferSizeMV))
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &minus_one, matA, d_X.vec, &one, d_R.vec,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))
      
        
       
        //--------------------------------------------------------------------------
        // ### 2 ### R_i_aux = L^-1 L^-T R_i
        size_t bufferSizeL, bufferSizeLT;
    void* d_bufferL, * d_bufferLT;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrLT;
    //    (a) L^-T tmp => R_i_aux    (triangular solver)
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrLT))
        CHECK_CUSPARSE(cusparseSpSV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
            &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, &bufferSizeLT))
        CHECK_CUDA(cudaMalloc(&d_bufferLT, bufferSizeLT))
        CHECK_CUSPARSE(cusparseSpSV_analysis(
            cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
            &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, d_bufferLT))
        CHECK_CUDA(cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double)))
        CHECK_CUSPARSE(cusparseSpSV_solve(
            cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
            &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT))

        //    (b) L^-T R_i => tmp    (triangular solver)
        CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescrL))
        CHECK_CUSPARSE(cusparseSpSV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL))
        CHECK_CUDA(cudaMalloc(&d_bufferL, bufferSizeL))
        CHECK_CUSPARSE(cusparseSpSV_analysis(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL))
        CHECK_CUDA(cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)))
        CHECK_CUSPARSE(cusparseSpSV_solve(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL))
        
        //--------------------------------------------------------------------------
        // ### 3 ### P0 = R0_aux
        CHECK_CUDA(cudaMemcpy(d_P.ptr, d_R_aux.ptr, m * sizeof(double),
            cudaMemcpyDeviceToDevice))
        //--------------------------------------------------------------------------
        // nrm_R0 = ||R||
        double nrm_R;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
        PRINT_INFO(nrm_R)
        double threshold = tolerance;// *nrm_R;
    printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    double delta;
    CHECK_CUBLAS(cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R.ptr, 1, &delta))
        //--------------------------------------------------------------------------
        // ### 4 ### repeat until convergence based on max iterations and
        //           and relative residual
        for (int i = 0; i < maxIterations; i++) {
            printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
            //----------------------------------------------------------------------
            // ### 5 ### alpha = (R_i, R_aux_i) / (A * P_i, P_i)
            //     (a) T  = A * P_i
            CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                matA, d_P.vec, &zero, d_T.vec,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                d_bufferMV))
          
                //     (b) denominator = (T, P_i)
                double denominator;
            CHECK_CUBLAS(cublasDdot(cublasHandle, m, d_T.ptr, 1, d_P.ptr, 1,
                &denominator))
                //     (c) alpha = delta / denominator
                double alpha = delta / denominator;
            PRINT_INFO(delta)
                PRINT_INFO(denominator)
                PRINT_INFO(alpha)
                //----------------------------------------------------------------------
                // ### 6 ###  X_i+1 = X_i + alpha * P
                //    (a) X_i+1 = -alpha * T + X_i
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, m, &alpha, d_P.ptr, 1,
                    d_X.ptr, 1))
                //----------------------------------------------------------------------
                // ### 7 ###  R_i+1 = R_i - alpha * (A * P)
                //    (a) R_i+1 = -alpha * T + R_i
                double minus_alpha = -alpha;
            CHECK_CUBLAS(cublasDaxpy(cublasHandle, m, &minus_alpha, d_T.ptr, 1,
                d_R.ptr, 1))
                //----------------------------------------------------------------------
                // ### 8 ###  check ||R_i+1|| < threshold
                CHECK_CUBLAS(cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
                PRINT_INFO(nrm_R)
                if (nrm_R < threshold)
                    break;
            //----------------------------------------------------------------------
            // ### 9 ### R_aux_i+1 = L^-1 L^-T R_i+1
            //    (a) L^-T R_i+1 => tmp    (triangular solver)
            CHECK_CUDA(cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double)))
                CHECK_CUDA(cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)))
                CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, matL, d_R.vec, d_tmp.vec,
                    CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvDescrL))
                //    (b) L^-T tmp => R_aux_i+1    (triangular solver)
                CHECK_CUSPARSE(cusparseSpSV_solve(cusparseHandle,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &one, matL, d_tmp.vec,
                    d_R_aux.vec, CUDA_R_64F,
                    CUSPARSE_SPSV_ALG_DEFAULT,
                    spsvDescrLT))
                //----------------------------------------------------------------------
                // ### 10 ### beta = (R_i+1, R_aux_i+1) / (R_i, R_aux_i)
                //    (a) delta_new => (R_i+1, R_aux_i+1)
                double delta_new;
            CHECK_CUBLAS(cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1,
                &delta_new))
                //    (b) beta => delta_new / delta
                double beta = delta_new / delta;
           PRINT_INFO(delta_new)
                PRINT_INFO(beta)
                delta = delta_new;
            //----------------------------------------------------------------------
            // ### 11 ###  P_i+1 = R_aux_i+1 + beta * P_i
            //    (a) copy R_aux_i+1 in P_i
            CHECK_CUDA(cudaMemcpy(d_P.ptr, d_R_aux.ptr, m * sizeof(double),
                cudaMemcpyDeviceToDevice))
                //    (b) P_i+1 = beta * P_i + R_aux_i+1
                CHECK_CUBLAS(cublasDaxpy(cublasHandle, m, &beta, d_P.ptr, 1,
                    d_P.ptr, 1))
        }
    //--------------------------------------------------------------------------
    printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    CHECK_CUDA(cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
        cudaMemcpyDeviceToDevice))
        // R = -A * X + R

        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
            matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV))
        // check ||R||
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R))
        printf("Final error norm = %e\n", nrm_R);
    double* x = new double[m];
    cout << "x:\n";
    CHECK_CUDA(cudaMemcpy(x, d_X.ptr, m * sizeof(double),
        cudaMemcpyDeviceToHost))

        for (int i = 0; i < 20; ++i)
        {
            cout << x[i] << "\n";
        }
    delete[]x;
    //--------------------------------------------------------------------------
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrL))
        CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescrLT))
        CHECK_CUDA(cudaFree(d_bufferL))
        CHECK_CUDA(cudaFree(d_bufferLT))
        CHECK_CUDA(cudaFree(d_bufferMV))
        return EXIT_SUCCESS;
}
