# README.md

## Analysing the performance differences between cusparse and cuBLAS libraries for gemm.

GEMM, general matrix multiplication is used frequently by many machine learning algorithms.
The Nvidia libraries include several implementations and I wanted to see how these implementations
performed compared with one another.

For cusparse, I was in particular interested in 2-4 sparsity (having heard and read a little about
it.)  2-4 sparsity is a sparse matrix where two out of every group of 4 elements is non zero.  In our
case, I'm interested in column-wise groupings.  That is column elements 0-3, 4-7, 8-11 etc.

It was an interesting challenge creating sparse matrices with this pattern.  Check out 
generate_2_4_sparse_float_matrix_columnwise() and generate_2_4_sparse_float_matrix_rowwise().
I created two versions because it wasn't clear which version I'd need and if I'd have access to a
transpose function.

# Requirements to run this lab:
- Install cuda 12.8. https://developer.nvidia.com/cuda-downloads
- This will not work with cuda 11.*.
- I couldn't install 12.8 on coursera's cloud instance so I created a local instance on my Windows Laptop (Asus Zepherus GA4011)
- My system
    * Windows 11 Home 64-bit
    * Processor: AMD Ryzen 9 4900HS
    * GPU: NVIDIA GeForce RTX2060 with Max-Q
    * Memory: 16GB


# Usage
There are two programs:

1. matrix_perf - takes in 7 args (all required)
   cublas|cusparse - you must pick the algorithm
   M - rows in sparse matrix A
   K - cols in sparse matrix A, rows in dense matrix B
   N - cols in dense matrix B
   iterations - number of matrix ops to run and measure duration.
   t|f - debug mode (true, generates some debug output, like A, B and C)

   It generates a line like:
   cublasSgemm, 8, 8, 8, 1000, 10.0, .01

   which corresponds to <algorithm>, M, K, N, iterations, total_runtime(ms), average_runtime(ms)

2. test_common - invoke this without args to test the performance of cusparseSpMM and cublasSgemm.
There are two aspects we'd like to test. Comparing the performance of both on square matrices
that bound powers of 2 sizes (e.g. 3,4,6 (where 3 and 6 surround 4)).  And verifying that performance increases exponentially as the square size quadruples.

# Analysis 

# Notes

Insight #1: cuda 12.* has deprecated many of the older APIs used in this class.  When you look at
my code in matrix_common.cu (look at the earlier history in my github), you'll see me going to lots 
of pain trying to get the new API to play with the older examples from the class.  Basically, I
needed to rewrite the basic examples for cusparse matrix x matrix GEMM.

Insight #2: don't take short cuts when creating sparse matrices from dense matrices.  I tried
creating a sparse matrix from the dense matrix using a cusparseDense2sparse function.  Unluckily,
that function was deprecated in 12.*.  Next, I tried taking a short cut and not calling the cusparseDenseToSparse_bufferSize and cusparseDenseToSparse_analysis functions.  They are required.
You can't just call the cusparseDenseToSparse_convert function! (It really wasn't that obvious, but
in hindsight, it kinda makes sense that you need to call all three in order.)

Insight #3: make use of a function like printMatrixFromDevice() to verify what the the data looks
like on the device.  It took me way too long to realize that the cusparseDenseToSparse_convert
function doesn't do the right thing if you don't call _bufferSize and _analysis first!  

Insight #4: if you compile with '-g' to get symbols for gdb debugging, you still can't see
cudaMalloc'ed memory.  You'll only see '0x00'.  You need to copy that memory back to host to view it.
Hence insight #3. But still, gdb is pretty good when you want to look at host memory and stepping
through code.  Highly recommended.

Insight #5:  I really hate const char*.  Really painful getting errors like:
void foo(char *arg) {
    ...
}
...
  foo("bar");
Telling me that I need a const char.  Also took me too long to figure out that its easy to fix by
changing the function signature to:

void foo(const char *arg) {

}

Insight #6: I had problems running:
    sparseTest(7, 7, 7, 1000, false);
    sparseTest(8, 8, 8, 1000, false);
    sparseTest(9, 9, 9, 1000, false); /* this line fails in malloc phase. */
malloc() had problems generating a 9*9*sizeof(float) chunk of memory.  It gave "malloc() invalid 
size (unsorted)".  To get around this problem, I use (10, 10, 10) instead.  Note though, I could run
it successfully when using valgrid... but could not run it without generating the malloc() error.






