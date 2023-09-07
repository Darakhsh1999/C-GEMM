// "g++ -O3 -march=native -ffast-math -o matmul matmul.cpp"
// vector instructions - https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avxnewtechs=AVX2

#include <iostream>
#include <chrono>
#include <cassert>
#include <intrin.h>
#include <immintrin.h>

// User constants
const int ITER = 1;
const int N = 1024;

// Constants
#define REG_SIZE 8
#define ALIGN_SIZE 32

// Optimization tweaking
#define BLOCK_I 4
#define BLOCK_J 2
#define BLOCK_VEC_I 8
#define BLOCK_VEC_J 1
/*
    ITER = 1, N = 1024
    Best found tweaking parameters that dont give compile error (due to missing registers?) 
    For non vectorized matmuls (BLOCK_I, BLOCK_J) = (4,2) gives best performance
    vectorized_matmul: (2,)
    vectorized_matmul_transposed: (4,2)
    vectorized_matmul_transposed_v2: (4,2)
    vectorized_matmul_optimal: (8,1)
*/

float A[N*N] __attribute__ ((aligned (ALIGN_SIZE)));
float B[N*N] __attribute__ ((aligned (ALIGN_SIZE)));;
float BT[N*N] __attribute__ ((aligned (ALIGN_SIZE)));
float BW[N*N] __attribute__ ((aligned (ALIGN_SIZE)));;
float C[N*N] __attribute__ ((aligned (ALIGN_SIZE)));
float C_val[N*N] __attribute__ ((aligned (ALIGN_SIZE))); // validation matrix

// Matrix multiplication declarations 
void assert_matmul(std::string algorithm_name, float *P, float *Q);
void matmul_naive(float *A, float *B, float *C, int N);
void matmul(float *A, float *B, float *C, int N);
void matmul_transposed(float *A, float *B, float *C, int N);
void block_matmul(float *A, float *B, float *C, int N);
void block_matmul_transposed(float *A, float *B, float *C, int N);
void block_matmul_cached(float *A, float *B, float *C, int N);
void block_matmul_cached_transposed(float *A, float *B, float *C, int N);
void vectorized_matmul(float *A , float *B, float *C, int N); 
void vectorized_matmul_transposed(float *A , float *B, float *C, int N); 
void vectorized_matmul_transposed_v2(float *A , float *B, float *C, int N); 
void vectorized_matmul_optimal(float *A , float *B, float *C, int N);

int main() {
    bool check_assertions = true;
    bool call_vectorized_matmuls = false;

    ///// INITIALIZE /////

    // Initialize timing variables
    std::chrono::time_point<std::chrono::steady_clock> t0_s, t1_s, t2_s, t3_s, t4_s, t5_s, t6_s, t7_s, t8_s, t9_s, t10_s;
    std::chrono::time_point<std::chrono::steady_clock> t0_e, t1_e, t2_e, t3_e, t4_e, t5_e, t6_e, t7_e, t8_e, t9_e, t10_e;
    std::chrono::time_point<std::chrono::steady_clock> t_wizzle_s, t_wizzle_e;

    // Initialize matrices with random values 
    for (int q = 0; q < N; q++) {
        for (int l = 0; l < N; l++) {
            float a = (float) rand() / (float) RAND_MAX;
            float b = (float) rand() / (float) RAND_MAX;
            A[q*N + l] = a;
            B[q*N + l] = b;
            BT[l*N + q] = b;
            C[q*N + l] = 0;
        }
    }

    // Pre-process
    t_wizzle_s = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i+= REG_SIZE) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < REG_SIZE; k++) {
                BW[i*N + j*8 + k] = BT[(i+k)*N + j];
            }
        }
    }
    t_wizzle_e = std::chrono::steady_clock::now();

    ///// MATMULS /////

    // Ground truth 
    matmul(A, B, C_val, N);

    // Naive matmul
    if (ITER != 1) matmul_naive(A, B, C, N); // warm up
    t0_s = std::chrono::steady_clock::now();
    for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
        matmul_naive(A, B, C, N);
    }
    t0_e = std::chrono::steady_clock::now();
    if (check_assertions && ITER == 1) assert_matmul("Naive", C, C_val);

    // Standard matmul
    matmul(A, B, C, N); // warm up
    t1_s = std::chrono::steady_clock::now();
    for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
        matmul(A, B, C, N);
    }
    t1_e = std::chrono::steady_clock::now();
    if (check_assertions) assert_matmul("Matmul", C, C_val);

    // Transposed matmul
    matmul_transposed(A, BT, C, N); // warmup
    t2_s = std::chrono::steady_clock::now();
    for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
        matmul_transposed(A, BT, C, N);
    }
    t2_e = std::chrono::steady_clock::now();
    if (check_assertions) assert_matmul("Transposed", C, C_val);

    // Block matmul
    block_matmul(A, B, C, N); // warmup
    t3_s = std::chrono::steady_clock::now();
    for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
        block_matmul(A, B, C, N);
    }
    t3_e = std::chrono::steady_clock::now();
    if (check_assertions) assert_matmul("Block", C, C_val);

    // Block matmul transposed
    block_matmul_transposed(A, BT, C, N); // warmup
    t4_s = std::chrono::steady_clock::now();
    for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
        block_matmul_transposed(A, BT, C, N);
    }
    t4_e = std::chrono::steady_clock::now();
    if (check_assertions) assert_matmul("Block transposed", C, C_val);

    // Block matmul cached
    block_matmul_cached(A, B, C, N); // warmup
    t5_s = std::chrono::steady_clock::now();
    for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
        block_matmul_cached(A, B, C, N);
    }
    t5_e = std::chrono::steady_clock::now();
    if (check_assertions) assert_matmul("Block cached", C, C_val);

    // Block matmul cached transposed
    block_matmul_cached_transposed(A, BT, C, N); // warmup
    t6_s = std::chrono::steady_clock::now();
    for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
        block_matmul_cached_transposed(A, BT, C, N);
    }
    t6_e = std::chrono::steady_clock::now();
    if (check_assertions) assert_matmul("Block cached transposed", C, C_val);

    ///// VECTORIZED MATMULS /////
    if (call_vectorized_matmuls) {

        // Vectorized matmul
        t7_s = std::chrono::steady_clock::now();
        for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
            vectorized_matmul(A, B, C, N);
        }
        t7_e = std::chrono::steady_clock::now();
        if (check_assertions) assert_matmul("Vectorized", C, C_val);

        // Vectorized matmul transposed
        t8_s = std::chrono::steady_clock::now();
        for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
            vectorized_matmul_transposed(A, BT, C, N);
        }
        t8_e = std::chrono::steady_clock::now();
        if (check_assertions) assert_matmul("Vectorized transposed", C, C_val);

        // Vectorized matmul transposed V2
        t9_s = std::chrono::steady_clock::now();
        for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
            vectorized_matmul_transposed_v2(A, BT, C, N);
        }
        t9_e = std::chrono::steady_clock::now();
        if (check_assertions) assert_matmul("Vectorized transposed V2", C, C_val);

        // Optimal vectorized matmul
        t10_s = std::chrono::steady_clock::now();
        for (int iter_idx = 0; iter_idx < ITER; iter_idx++){
            vectorized_matmul_optimal(A, BW, C, N);
        }
        t10_e = std::chrono::steady_clock::now();
        if (check_assertions) assert_matmul("Optimal", C, C_val);

    } 


    ///// TIMINGS /////

    // Calculate flops
    float flop = (2.0*N*N*N)*1e-9; // GFLOP
    float s_twizzle, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;
    s_twizzle = (std::chrono::duration_cast<std::chrono::nanoseconds> (t_wizzle_e - t_wizzle_s).count())*1e-9/float(ITER);
    s0 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t0_e - t0_s).count())*1e-9/float(ITER);
    s1 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t1_e - t1_s).count())*1e-9/float(ITER);
    s2 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t2_e - t2_s).count())*1e-9/float(ITER);
    s3 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t3_e - t3_s).count())*1e-9/float(ITER);
    s4 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t4_e - t4_s).count())*1e-9/float(ITER);
    s5 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t5_e - t5_s).count())*1e-9/float(ITER);
    s6 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t6_e - t6_s).count())*1e-9/float(ITER);
    s7 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t7_e - t7_s).count())*1e-9/float(ITER);
    s8 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t8_e - t8_s).count())*1e-9/float(ITER);
    s9 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t9_e - t9_s).count())*1e-9/float(ITER);
    s10 = (std::chrono::duration_cast<std::chrono::nanoseconds> (t10_e - t10_s).count())*1e-9/float(ITER);
    float flops0, flops1, flops2, flops3, flops4, flops5, flops6, flops7, flops8, flops9, flops10; 
    flops0 = flop/s0;
    flops1 = flop/s1;
    flops2 = flop/s2;
    flops3 = flop/s3;
    flops4 = flop/s4;
    flops5 = flop/s5;
    flops6 = flop/s6;
    flops7 = flop/s7;
    flops8 = flop/s8;
    flops9 = flop/s9;
    flops10 = flop/s10;

    ///// PRINTING /////

    // Print benchmarks
    std::cout << "All tests passed!" << std::endl << std::endl;
    std::cout << "Time = " << s_twizzle*1e3 << " [ms]: Twizzle" << std::endl;
    std::cout << "Time = " << s0*1e3 << " [ms]: Naive" << std::endl;
    std::cout << flops0 << " GFLOPS" << std::endl;
    std::cout << "Time = " << s1*1e3 << " [ms]: Standard" << std::endl;
    std::cout << flops1 << " GFLOPS" << std::endl;
    std::cout << "Time = " << s2*1e3 << " [ms]: Transposed" << std::endl;
    std::cout << flops2 << " GFLOPS" << std::endl;
    std::cout << "Time = " << s3*1e3 << " [ms]: Block" << std::endl;
    std::cout << flops3 << " GFLOPS" << std::endl;
    std::cout << "Time = " << s4*1e3 << " [ms]: Block transposed" << std::endl;
    std::cout << flops4 << " GFLOPS" << std::endl;
    std::cout << "Time = " << s5*1e3 << " [ms]: Block cached" << std::endl;
    std::cout << flops5 << " GFLOPS" << std::endl;
    std::cout << "Time = " << s6*1e3 << " [ms]: Block cached transposed" << std::endl;
    std::cout << flops6 << " GFLOPS" << std::endl;
    if (call_vectorized_matmuls) {
        std::cout << "Time = " << s7*1e3 << " [ms]: Vectorized" << std::endl;
        std::cout << flops7 << " GFLOPS" << std::endl;
        std::cout << "Time = " << s8*1e3 << " [ms]: Vectorized transposed" << std::endl;
        std::cout << flops8 << " GFLOPS" << std::endl;
        std::cout << "Time = " << s9*1e3 << " [ms]: Vectorized transposed V2" << std::endl;
        std::cout << flops9 << " GFLOPS" << std::endl;
        std::cout << "Time = " << s10*1e3 << " [ms]: Optimal " << std::endl;
        std::cout << flops10 << " GFLOPS" << std::endl;
    }

    return 0;
}



// Used to assert correctness for matmul
void assert_matmul(std::string algorithm_name, float *P, float *Q) {
    float p;
    float q;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            p = P[i*N + j];
            q = Q[i*N + j];
            if (abs(p-q) > 1e-3) {
                std::cout << algorithm_name << ": mismatch at [i,j] = [" << i << "," << j << "] " << p << " != " << q << std::endl;
                exit(-1);
            }
        }
    }
    std::cout << "Test for " << algorithm_name << " passed" << std::endl;
}


// Naive matmul (slow)
void matmul_naive(float *A, float *B, float *C, int N) {
    /* 
    28.91 GFLOPS @ 1024x1024 (4,2)
    Naive implementation without having a accumulate variable which means 
    we access the C matrix N times for each value.
    */

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}


// Standard matmul
void matmul(float *A, float *B, float *C, int N) {
    /* 
    29.65 GFLOPS @ 1024x1024 (4,2)
    Standard matmul implementation where we store the vector dot product in an
    accumulating variable so we only access the C matrix once for each value.
    */

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0;
            for (int k = 0; k < N; k++) {
                acc += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = acc;
        }
    }
}


// Transposed matmul (assume B is transposed)
void matmul_transposed(float *A, float *B, float *C, int N) {
    /* 
    39.19 GFLOPS @ 1024x1024 (4,2)
    Standard matmul implementation where we access both A and B matrix in row 
    major order making is faster since row major access is faster than column
    major order.
    */

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0;
            for (int k = 0; k < N; k++) {
                acc += A[i*N + k] * B[j*N + k];
            }
            C[i*N + j] = acc;
        }
    }
}


// Block matmul
void block_matmul(float *A, float *B, float *C, int N) {
    /*
    1.74 GFLOPS @ 1024x1024 (4,2)
    Calculating the elements in C in blocks of size (BLOCK_I, BLOCK_J) 
    where iteration inside each block is also in row major order.
    */

    assert(N % BLOCK_I == 0); // even block matrices
    assert(N % BLOCK_J == 0); // even block matrices

    for (int bi = 0; bi < N; bi += BLOCK_I) {
        for (int bj = 0; bj < N; bj += BLOCK_J) {

            for (int i = 0; i < BLOCK_I; i++) {
                for (int j = 0; j < BLOCK_J; j++) {
                    float acc = 0.0;
                    for (int k = 0; k < N; k++) {
                        acc += A[(bi+i)*N + k] * B[k*N + (bj+j)];
                    }
                    C[(bi+i)*N + (bj+j)] = acc;
                }
            }

        }
    }
}


// Block matmul transposed (assume B is transposed)
void block_matmul_transposed(float *A, float *B, float *C, int N) {
    /*
    51.37 GFLOPS @ 1024x1024 (4,2)
    Block wise matmul where the access to B matrix is in row major order
    */


    assert(N % BLOCK_I == 0); // even block matrices
    assert(N % BLOCK_J == 0); // even block matrices

    for (int bi = 0; bi < N; bi += BLOCK_I) {
        for (int bj = 0; bj < N; bj += BLOCK_J) {

            for (int i = 0; i < BLOCK_I; i++) {
                for (int j = 0; j < BLOCK_J; j++) {
                    float acc = 0.0;
                    for (int k = 0; k < N; k++) {
                        acc += A[(bi+i)*N + k] * B[(bj+j)*N + k];
                    }
                    C[(bi+i)*N + (bj+j)] = acc;
                }
            }
        }
    }
}


// Block matmul cached
void block_matmul_cached(float *A, float *B, float *C, int N) {
    /*
    9.24 GFLOPS @ 1024x1024 (4,2)
    Block matmul where the i'th multiplication in the 
    vector dot product is done for each element in the block before calculating the
    next multiplication element. This keeps memory access more localized 
    compared to regular block matul.
    */

    assert(N % BLOCK_I == 0); // even block matrices
    assert(N % BLOCK_J == 0); // even block matrices

    for (int bi = 0; bi < N; bi += BLOCK_I) {
        for (int bj = 0; bj < N; bj += BLOCK_J) {

            // Compute
            float c_block[BLOCK_I*BLOCK_J] = {};
            for (int k = 0; k < N; k++) {
                for (int i = 0; i < BLOCK_I; i++) {
                    for (int j = 0; j < BLOCK_J; j++) {
                        c_block[i*BLOCK_J + j] += A[(bi+i)*N + k] * B[k*N + (bj+j)];
                    }
                }
            }

            // Store
            for (int i = 0; i < BLOCK_I; i++) {
                for (int j = 0; j < BLOCK_J; j++) {
                    C[(bi+i)*N + (bj+j)] = c_block[i*BLOCK_J + j];
                }
            }

        }
    }
}

// Block matmul cached
void block_matmul_cached_transposed(float *A, float *B, float *C, int N) {
    /*
    113.24 GFLOPS @ 1024x1024 (4,2)
    Block matmul with row major access to the B matrix.
    The most optimal algorithm before introducing intrinsics
    */

    assert(N % BLOCK_I == 0); // even block matrices
    assert(N % BLOCK_J == 0); // even block matrices

    for (int bi = 0; bi < N; bi += BLOCK_I) {
        for (int bj = 0; bj < N; bj += BLOCK_J) {

            // Compute
            float c_block[BLOCK_I*BLOCK_J] = {};
            for (int k = 0; k < N; k++) {
                for (int i = 0; i < BLOCK_I; i++) {
                    for (int j = 0; j < BLOCK_J; j++) {
                        c_block[i*BLOCK_J + j] += A[(bi+i)*N + k] * B[(bj+j)*N + k];
                    }
                }
            }

            // Store
            for (int i = 0; i < BLOCK_I; i++) {
                for (int j = 0; j < BLOCK_J; j++) {
                    C[(bi+i)*N + (bj+j)] = c_block[i*BLOCK_J + j];
                }
            }

        }
    }
}

// Vectorized matmul
void vectorized_matmul(float *A , float *B, float *C, int N) {
    /*
    14.70 GFLOPS @ 1024x1024
    Parallelizes the vector dot product for each row in the block
    with size (BLOCK_I, REG_SIZE=8). This requires broadcasting each
    */

    assert(N % BLOCK_VEC_I == 0); // even block matrices

    __m256 *Bm = (__m256*)B;
    __m256 *Cm = (__m256*)C;

    for (int bi = 0; bi < N; bi += BLOCK_VEC_I) {
        for (int bj = 0; bj < N; bj += REG_SIZE) {

            // Compute
            __m256 vec_block[BLOCK_VEC_I];
            for (int i = 0; i < BLOCK_VEC_I; i++) {

                __m256 vec_row = {};
                for (int k = 0; k < N; k++) {
                    __m256 a_vec = _mm256_broadcast_ss(&A[(bi+i)*N + k]); // fill register with one value
                    vec_row = _mm256_fmadd_ps(
                        a_vec,
                        Bm[(k*N + bj)/REG_SIZE],
                        vec_row);

                }
                vec_block[i] = vec_row;
            }

            // Store
            for (int i = 0; i < BLOCK_VEC_I; i++) {
                for (int j = 0; j < REG_SIZE; j++) {
                    Cm[((bi+i)*N + bj+j)/REG_SIZE] = vec_block[i];
                }
            }
        }
    }
}

void vectorized_matmul_transposed(float *A , float *B, float *C, int N) {
    /*
    18.96 GFLOPS @ 1024x1024
    Vectorized version of block_matmul_transposed() where we jump in 
    increments of REG_SIZE in the vector dot product and reduce over
    the register at the end to get the dot product sum.
    */

    assert(N % BLOCK_VEC_I == 0); // even block matrices
    assert(N % BLOCK_VEC_J == 0); // even block matrices
    __m256 *Am = (__m256*)A;
    __m256 *Bm = (__m256*)B;

    for (int bi = 0; bi < N; bi += BLOCK_VEC_I) {
        for (int bj = 0; bj < N; bj += BLOCK_VEC_J) {

            // Compute
            float c_block[BLOCK_VEC_I*BLOCK_VEC_J];
            for (int i = 0; i < BLOCK_VEC_I; i++) {
                for (int j = 0; j < BLOCK_VEC_J; j++) {
                    __m256 c_vec = {};
                    for (int k = 0; k < N; k += REG_SIZE) {
                        c_vec = _mm256_fmadd_ps(
                            Am[((bi+i)*N + k)/REG_SIZE],
                            Bm[((bj+j)*N + k)/REG_SIZE],
                            c_vec);
                    }

                    // Store
                    float ftmp = 0.0;
                    for (int q = 0; q < REG_SIZE; q++) ftmp += c_vec[q];
                    c_block[i*BLOCK_VEC_J + j] = ftmp;
                }
            }

            // Write
            for (int i = 0; i < BLOCK_VEC_I; i++) {
                for (int j = 0; j < BLOCK_VEC_J; j++) {
                    C[(bi+i)*N + (bj+j)] = c_block[i*BLOCK_VEC_J + j];
                }
            }
        }
    }
}


void vectorized_matmul_transposed_v2(float *A , float *B, float *C, int N) {
    /*
    43.96 GFLOPS @ 1024x1024
    */

    assert(N % BLOCK_VEC_I == 0); // even block matrices
    assert(N % BLOCK_VEC_J == 0); // even block matrices
    __m256 *Am = (__m256*)A;
    __m256 *Bm = (__m256*)B;
    __m256 *Cm = (__m256*)C;

    for (int bi = 0; bi < N; bi += BLOCK_VEC_I) {
        for (int bj = 0; bj < N; bj += BLOCK_VEC_J) {

            __m256 c_block[BLOCK_VEC_I*BLOCK_VEC_J] = {};
            for (int k = 0; k < N; k += REG_SIZE) {

                for (int i = 0; i < BLOCK_VEC_I; i++) {
                    for (int j = 0; j < BLOCK_VEC_J; j++) {
                            c_block[i*BLOCK_VEC_J + j] = _mm256_fmadd_ps(
                                Am[((bi+i)*N + k)/REG_SIZE],
                                Bm[((bj+j)*N + k)/REG_SIZE],
                                c_block[i*BLOCK_VEC_J + j]);
                    }
                }
            }

            // Store
            for (int i = 0; i < BLOCK_VEC_I; i++) {
                for (int j = 0; j < BLOCK_VEC_J; j++) {
                    float ftmp = 0.0;
                    for (int q = 0; q < REG_SIZE; q++) ftmp += c_block[i*BLOCK_VEC_J + j][q];
                    C[((bi+i)*N + (bj+j))] = ftmp;
                }
            }
        }
    }
}


void vectorized_matmul_optimal(float *A , float *B, float *C, int N) {
    /*
    140.96 GFLOPS @ 1024x1024
    */

    assert(N % BLOCK_VEC_I == 0); // even block matrices
    assert(N % BLOCK_VEC_J == 0); // even block matrices
    __m256 *Cm = (__m256*)C;
    __m256 *Bm = (__m256*)B;

    for (int bi = 0; bi < N; bi += BLOCK_VEC_I) {
        for (int bj = 0; bj < N; bj += REG_SIZE*BLOCK_J) {

            __m256 c_block[BLOCK_VEC_I*BLOCK_VEC_J] = {};
            for (int k = 0; k < N; k++) {

                for (int i = 0; i < BLOCK_VEC_I; i++) {
                    __m256 a_vec = _mm256_broadcast_ss(&A[(bi+i)*N + k]);
                    for (int j = 0; j < BLOCK_VEC_J; j++) {
                            c_block[i*BLOCK_VEC_J + j] = _mm256_fmadd_ps(
                                a_vec,
                                Bm[((bj+j*REG_SIZE)*N + REG_SIZE*k)/REG_SIZE],
                                c_block[i*BLOCK_VEC_J + j]);
                    }
                }
            }

            // Store
            for (int i = 0; i < BLOCK_VEC_I; i++) {
                for (int j = 0; j < BLOCK_VEC_J; j++) {
                    Cm[((bi+i)*N + bj + j*REG_SIZE)/REG_SIZE] = c_block[i*BLOCK_VEC_J + j];
                }
            }
        }
    }
}