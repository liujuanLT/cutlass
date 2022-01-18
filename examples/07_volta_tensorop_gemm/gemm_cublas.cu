#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <assert.h>
#include "cutlass/util/command_line.h"

#ifndef CHECK_ARCH
#define CHECK_ARCH 1
#endif

#define LOG_LEVEL 1

using namespace std;

#ifndef DATA_TYPE
#define DATA_TYPE 1  // 0: float, 1: half, 1:int8
#endif

#if DATA_TYPE == 0
    using datatype = float;
#elif DATA_TYPE == 1
    using datatype = __half;
#else
    using datatype = int8_t;
#endif

void init_vector(float* d_data, int len) {
    // a rough piece of code considering no performance 
    float* h_data = (float*) malloc(sizeof(float) * len);
    for (size_t i = 0; i < len; ++ i) {
        h_data[i] = (float) (i + 1);
    }
    cudaMemcpy(d_data, h_data, sizeof(float) * len, cudaMemcpyHostToDevice);
    free(h_data);
}

void init_vector(__half* d_data, int len) {
    // a rough piece of code considering no performance 
    __half* h_data = (__half*) malloc(sizeof(__half) * len);
    for (size_t i = 0; i < len; ++ i) {
        h_data[i] = __float2half(1);
    }
    cudaMemcpy(d_data, h_data, sizeof(__half) * len, cudaMemcpyHostToDevice);
    free(h_data);
}

void print_matrix(const float* d_mat, int m, int n) {
    // a rough piece of code considering no performance 
    float* h_mat = (float*) malloc(sizeof(float) * m * n);
    cudaMemcpy(h_mat, d_mat, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    for (size_t irow = 0; irow < min(m, 10); ++ irow) {
        for (size_t icol = 0; icol < min(n, 10); ++ icol) {
            std::cout << h_mat[n*irow + icol] << " " ;
        }
        std::cout << std::endl;
    }
    free(h_mat);
}

void print_matrix(const __half* d_mat, int m, int n) {
    // a rough piece of code considering no performance 
    __half* h_mat = (__half*) malloc(sizeof(__half) * m * n);
    cudaMemcpy(h_mat, d_mat, sizeof(__half) * m * n, cudaMemcpyDeviceToHost);
    for (size_t irow = 0; irow < min(m, 10); ++ irow) {
        for (size_t icol = 0; icol < min(n, 10); ++ icol) {
            std::cout << __half2float(h_mat[n*irow + icol]) << " " ;
        }
        std::cout << std::endl;
    }
    free(h_mat);
}

void warmUp ()
{
  const int N = 1000;

  // init stream and cublas
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle); // takes hundrads of milliseconds for first time
  
  // cudaMemcpy
  float* h_data = (float*) malloc(sizeof(float) * N);;
  for (int i = 0; i < N; ++ i) h_data[i] = 1.0f * i;
  float* d_data(NULL);
  cudaMalloc((void**)&d_data, sizeof(float) * N);
  cudaMemcpy(d_data, h_data, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(h_data, d_data, sizeof(float) * N, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++ i) assert(h_data[i] == 1.0f * i);
  free(h_data);
  cudaFree(d_data);
}

int run(int m, int k, int n, uint64_t niters) {
   
    cout <<"m,k,n: " << m << ", " << k << "," << n << endl;
    cout << "datatype: " << DATA_TYPE << endl;

    datatype *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, sizeof(datatype) * m * k);
    cudaMalloc((void**) &d_B, sizeof(datatype) * k * n);
    cudaMalloc((void**) &d_C, sizeof(datatype) * m * n);
    cudaMemset(d_C, 0, sizeof(datatype) * m * n);
    init_vector(d_A, m * k);
    init_vector(d_B, k * n);
    init_vector(d_A, m * n);

#if LOG_LEVEL >= 2
    cout << "A:" << endl;
    print_matrix(d_A, m, k);
    cout << "B:" << endl;
    print_matrix(d_B, k, n);
#endif
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    datatype alpha = 1.0f;
    datatype beta = 0.0f;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    int lda = (transA == CUBLAS_OP_N) ? m : k;
    int ldb = (transB == CUBLAS_OP_N) ? k : n;
    int ldc = n;

    cudaError_t ret;
    cudaEvent_t events[2];

    for (auto & event : events) {
      ret = cudaEventCreate(&event);
      if (ret != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(ret) << std::endl;
        return -1;
      }
    }
    
    // Record an event at the start of a series of GEMMs
    ret = cudaEventRecord(events[0]);
    if (ret != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(ret) << std::endl;
        return -1;
    }

    for (int iter = 0; iter < niters; ++iter) {
        #if DATA_TYPE == 0 // float
            cublasSgemm(handle, transA, transB, (int)m, (int)n, (int)k, 
                &alpha, d_A, (int)lda, d_B, (int)ldb, &beta, d_C, (int)ldc);
        #elif DATA_TYPE == 1 // half
            cublasHgemm(handle, transA, transB, (int)m, (int)n, (int)k, 
                &alpha, d_A, (int)lda, d_B, (int)ldb, &beta, d_C, (int)ldc);            
        #endif
      }

    ret = cudaEventRecord(events[1]);
    if (ret != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(ret) << std::endl;
    return -1;
    }      
       
    // Wait for work on the device to complete.
    ret = cudaEventSynchronize(events[1]);
    if (ret != cudaSuccess) {
        std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(ret) << std::endl;
        return -1;
    }

    float runtime_ms = 0;
    ret = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (ret != cudaSuccess) {
        std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(ret) << std::endl;
        return -1;
    }
    runtime_ms = double(runtime_ms) / double(niters);

    // Cleanup
    for (auto event : events) {
        (void)cudaEventDestroy(event);
    }

#if LOG_LEVEL >= 2
    cout << "C:" << endl;
    print_matrix(d_C, m, n);
#endif

    cout << "Runtime: " << runtime_ms << " ms" << endl;
    cout << "total time: " << runtime_ms * niters << " ms for " << niters << " loops" << endl;
    cout << "transA, transB: " << ((transA == CUBLAS_OP_N) ? "N" : "T") << " ,"
         << ((transB == CUBLAS_OP_N) ? "N" : "T") << endl;

    // release
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

int main(int argc, const char **argv) {
    #if CHECK_ARCH
        // Volta Tensor Core operations exposed with mma.sync are first available in CUDA 10.1.
        //
        // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
        if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
        std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;
        return 0;
        }

        cudaDeviceProp props;

        cudaError_t error = cudaGetDeviceProperties(&props, 0);
        if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
        }
    
        if (props.major != 7) {
        std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
                    << std::endl;
        // Return 0 so tests are considered passing if run on unsupported architectures or CUDA Toolkits.
        return 0;
        }
    #endif

    cutlass::CommandLine cmd(argc, argv);
    int m = 2, k = 3, n = 2;
    uint64_t niters = 1;
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("k", k);        
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("niters", niters);

    warmUp();

    run(m, k, n, niters);

    return 0;
  }