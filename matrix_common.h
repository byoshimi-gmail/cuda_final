
void generate_dense_float_matrix(int M, int N, float min_val, float max_val, float **outA);
void generate_2_4_sparse_float_matrix_rowwise(int M, int N, float min_val, float max_val, float **outA);
void generate_2_4_sparse_float_matrix_columnwise(int M, int N, float min_val, float max_val, float **outA);
void print_matrix(const char *name, float *M, int nrows, int ncols, int max_row,
                  int max_col);


void sparseTest(int M, int N, int K, int iterations, bool debug);
void cublasTest(int M, int N, int K, int iterations, bool debug);