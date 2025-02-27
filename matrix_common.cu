#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cassert>            // assert
#include <cusparse_v2.h>
#include <cuda.h>
#include "common/common.h"

// Common cuda functions wrapping matrix operations.

// Create matrix populated with random values in host memory.
// Column major order since this is what cusparse library expects.
void generate_dense_float_matrix(int M, int N, float min_val, float max_val, float **outA)
{
    float* out=(float*)malloc(M*N*sizeof(float));

    assert(max_val > min_val);

    for (int col=0; col<N; col++) {
        for (int row=0; row<M; row++) {
            *(out+row*N+col) = (max_val-min_val)*rand() / 2147483647 + min_val;
        }
    }
    *outA = out;
}

// 2-4 (two out of four elements are non-zero, rowwise)
void generate_2_4_sparse_float_matrix_rowwise(int M, int N, float min_val, float max_val, float **outA)
{
    float* out=(float*)malloc(M*N*sizeof(float));

    int masks[6] = {/*0011*/3, /*0101*/5, /*1001*/9, /*0110*/6, /*1010*/ 10, /*1100*/12}; /* 6 variants*/

    assert(max_val > min_val);

    for (int col=0; col<N; col++) {
        for (int row=0; row<M; row+=4) {
            // Min of at least 2 out of every 4 non-zero

            float rnd = ((float)rand() / 2147483647);
            int pick6 = rnd * 6.0;
            int mask=masks[pick6];

            *(out+row*N+col) = mask&0x01?(max_val-min_val)*rand() / 2147483647 + min_val : 0;
            *(out+(row+1)*N+col) = mask&0x02?(max_val-min_val)*rand() / 2147483647 + min_val : 0;
            *(out+(row+2)*N+col) = mask&0x04?(max_val-min_val)*rand() / 2147483647 + min_val : 0;
            *(out+(row+3)*N+col) = mask&0x08?(max_val-min_val)*rand() / 2147483647 + min_val : 0;

        }
    }
    *outA = out;
}


// 2-4 (two out of four elements are non-zero, columnwise
void generate_2_4_sparse_float_matrix_columnwise(int M, int N, float min_val, float max_val, float **outA)
{
    float* out=(float*)malloc(M*N*sizeof(float));

    int masks[6] = {/*0011*/3, /*0101*/5, /*1001*/9, /*0110*/6, /*1010*/ 10, /*1100*/12}; /* 6 variants*/

    assert(max_val > min_val);

    for (int row=0; row<N; row++) {
        for (int col=0; col<M; col+=4) {
            // Min of at least 2 out of every 4 non-zero

            float rnd = ((float)rand() / 2147483647);
            int pick6 = rnd * 6.0;
            int mask=masks[pick6];

            *(out+row*N+col) = mask&0x01?(max_val-min_val)*rand() / 2147483647 + min_val : 0;
            *(out+row*N+col+1) = mask&0x02?(max_val-min_val)*rand() / 2147483647 + min_val : 0;
            *(out+row*N+col+2) = mask&0x04?(max_val-min_val)*rand() / 2147483647 + min_val : 0;
            *(out+row*N+col+3) = mask&0x08?(max_val-min_val)*rand() / 2147483647 + min_val : 0;

        }
    }
    *outA = out;
}


// Print part of a matrix
void print_matrix(char *name, float *M, int nrows, int ncols, int max_row,
        int max_col)
{
    int row, col;

    printf("Dumping matrix %s: (max_rows=%d, max_cols=%d)\n", name, max_row, max_col);

    for (row = 0; row < max_row; row++)
    {
        for (col = 0; col < max_col; col++)
        {
            printf("%2.2f ", M[row * ncols + col]);
        }
        printf("...\n");
    }
    printf("...\n");
}

void sparseTest(int M, int N, int K) {
    float alpha=1;
    float beta=1;

    float *A = (float*)malloc(M*K*sizeof(float));
    float *B = (float*)malloc(K*N*sizeof(float));
    float *C = (float*)malloc(M*N*sizeof(float));

    generate_2_4_sparse_float_matrix_columnwise(M, N, 0.0, 10.0, &A);
    generate_dense_float_matrix(K, N, -1.0, 1.0, &B);
    C = (float *)malloc(sizeof(float) * M * M);

    // Create the cuSPARSE handle
    cusparseHandle_t handle = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Allocate device memory for vectors and the dense form of the matrix A
    float *dA, *dB, *dC;
    int *dNumZerosPerRowA;
    int totalANnz;


    CHECK(cudaMalloc((void **)&dNumZerosPerRowA, sizeof(int) * M));

    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * K));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * K * M));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dNumZerosPerRowA, sizeof(int) * M));

    // Construct a descriptor of the matrix A
    cusparseMatDescr_t Adescr = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&Adescr));
    CHECK_CUSPARSE(cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO));

    // Transfer the input vectors and dense matrix A to the device
    CHECK(cudaMemcpy(dA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0x00, sizeof(float) * M * N));

    // Compute the number of non-zero elements in A
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, K, Adescr,
                                dA, M, dNumZerosPerRowA, &totalANnz));

    /*
    if (totalANnz != trueANnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value: expected %d but got %d\n", trueANnz, totalANnz);
        return 1;
    }
    */

    // Allocate device memory to store the sparse CSR representation of A
    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;
    CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * totalANnz));
    CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalANnz));

    // Convert A from a dense formatting to a CSR formatting, using the GPU
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;



    CHECK_CUSPARSE(cusparseCreateCsr(&matA, M, K, totalANnz,
        dCsrRowPtrA, dCsrColIndA, dCsrValA,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // CHECK_CUSPARSE(cusparseSdense2csr(handle, M, K, Adescr, dA, M, dNumZerosPerRowA,
    //                                  dCsrValA, dCsrRowPtrA, dCsrColIndA,));

    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, K, N, K, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, M, N, M, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // Copy the result vector back to the host
    CHECK(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    char a_matrix_desc[]="A:";
    print_matrix(a_matrix_desc, A, M, K, 8, 8);
    char b_matrix_desc[]="b:";
    print_matrix(b_matrix_desc, B, K, N, 8, 8);
    char c_matrix_desc[]="C:";
    print_matrix(c_matrix_desc, C, M, N, 8, 8);

    free(A);
    free(B);
    free(C);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaFree(dNumZerosPerRowA));
    CHECK(cudaFree(dCsrValA));
    CHECK(cudaFree(dCsrRowPtrA));
    CHECK(cudaFree(dCsrColIndA));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(Adescr));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}