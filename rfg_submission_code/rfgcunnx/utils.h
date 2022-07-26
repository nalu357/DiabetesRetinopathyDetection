#ifndef CUNN_UTILS_H
#define CUNN_UTILS_H
#include <lua.h>
#include "THCGeneral.h"
THCState* getCutorchState(lua_State* L);

#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

//#define CHECK_EQ(a, b) assert((a) == (b))
#define CHECK_EQ(a, b) (a) != (b)

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
	if (error != cudaSuccess) {\
	fprintf(stderr, "error in cuda %s\n", cudaGetErrorString(error)); } \
	  \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#endif
