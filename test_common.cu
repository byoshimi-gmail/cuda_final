#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <unistd.h>
#include "matrix_common.h"

void test_generate_dense_float_matrix() {
  int m=20;
  int n=10;
  float minval=-10.0;
  float maxval=100.0;
  float *aMatrix=(float*)malloc(m*n*sizeof(float));

  generate_dense_float_matrix(m, n, minval, maxval, &aMatrix);
  char name[]="test 20x10 dense matrix";
  print_matrix(name, aMatrix, m, n, 10, 10);
}

void test_generate_2_4_sparse_float_matrix() {
  
  int m=8;
  int n=8;
  float minval=-10.0;
  float maxval=100.0;
  float *aMatrix=(float*)malloc(m*n*sizeof(float));

  generate_2_4_sparse_float_matrix_rowwise(m, n, minval, maxval, &aMatrix);
  char name[]="test 20x10 sparse matrix rowwise";
  print_matrix(name, aMatrix, m, n, 8, 8);

  generate_2_4_sparse_float_matrix_columnwise(m, n, minval, maxval, &aMatrix);
  char name2[]="test 20x10 sparse matrix columnwise";
  print_matrix(name2, aMatrix, m, n, 8, 8);
}

int main(void) {
   // test_generate_dense_float_matrix();

   // test_generate_2_4_sparse_float_matrix();
   printf("# algorithm, M, K, N, iterations, total_runtime(ms), average_runtime(ms)\n");
   int sizes[]={8, 16, 64, 256, 512, 1024, 2048, 4096};
   int runs = 8;

   for (int i=0; i<runs; i++) {
    int dim=sizes[i]-1;
    sparseTest(dim, dim, dim, 1000, false);
    cublasTest(dim, dim, dim, 1000, false);
    dim=sizes[i];
    sparseTest(dim, dim, dim, 1000, false);
    cublasTest(dim, dim, dim, 1000, false);
    dim=sizes[i]+2;
    sparseTest(dim, dim, dim, 1000, false);
    cublasTest(dim, dim, dim, 1000, false);
  }
   /*
   for (int i=0; i<runs; i++) {
    //sparseTest(sizes[i]-2, sizes[i]-2, sizes[i]-2, 1000, false);
    //sparseTest(sizes[i]-1, sizes[i]-1, sizes[i]-1, 1000, false);
    int dim=sizes[i]-1;
    sparseTest(dim, dim, dim, 1000, false);
    cublasTest(dim, dim, dim, 1000, false);
  }
   for (int i=0; i<runs; i++) {
     //sparseTest(sizes[i]-2, sizes[i]-2, sizes[i]-2, 1000, false);
     //sparseTest(sizes[i]-1, sizes[i]-1, sizes[i]-1, 1000, false);
     int dim=sizes[i];
     sparseTest(dim, dim, dim, 1000, false);
     cublasTest(dim, dim, dim, 1000, false);
   }
   for (int i=0; i<runs; i++) {
    //sparseTest(sizes[i]-2, sizes[i]-2, sizes[i]-2, 1000, false);
    //sparseTest(sizes[i]-1, sizes[i]-1, sizes[i]-1, 1000, false);
    int dim=sizes[i]+2;
    sparseTest(dim, dim, dim, 1000, false);
    cublasTest(dim, dim, dim, 1000, false);
  }
    */
}