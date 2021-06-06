#pragma once

#include <cstdio>
#include <cstdlib>

namespace cuFIXNUM {

/*
 * Convenience wrappers around some CUDA library functions
 */
static inline void
cuda_print_errmsg(cudaError err, const char *msg, const char *file, const int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal CUDA error at %s:%d : %s : %s\n",
                file, line, msg, cudaGetErrorString(err));
        if (cudaDeviceReset() != cudaSuccess)
            fprintf(stderr, "   ...and failed to reset the device!\n");
        exit(EXIT_FAILURE);
    }
}
}

#define cuda_check(err, msg)                            \
    ::cuFIXNUM::cuda_print_errmsg(err, msg, __FILE__, __LINE__)

namespace cuFIXNUM {

static inline int
cuda_get_cores_per_sm(int device_id=0) {
  cudaDeviceProp prop;
  cuda_check(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
  int cores = 0;
  switch (prop.major) {
     case 3: // Kepler
      cores = 192;
      break;
     case 5: // Maxwell
      cores = 128;
      break;
     case 6: // Pascal
      if ((prop.minor == 1) || (prop.minor == 2)) cores = 128;
      else if (prop.minor == 0) cores = 64;
      break;
     case 7: // Volta and Turing
      if ((prop.minor == 0) || (prop.minor == 5)) cores = 64;
      break;
     case 8: // Ampere
      if (prop.minor == 0) cores = 64;
      else if (prop.minor == 6) cores = 128;
      break;
     default:
      break;
  }
  return cores;
}

} // End namespace cuFIXNUM

#define cuda_malloc(ptr, size)                                  \
    cuda_check(cudaMalloc(ptr, size), "memory allocation")
#define cuda_malloc_managed(ptr, size)                                  \
    cuda_check(cudaMallocManaged(ptr, size),                            \
            "unified memory allocation (default attach)")
#define cuda_malloc_managed_host(ptr, size)                             \
    cuda_check(cudaMallocManaged(ptr, size, cudaMemAttachHost),         \
            "unified memory allocation (host attach)")
#define cuda_stream_attach_mem(stream, ptr)                             \
    cuda_check(cudaStreamAttachMemAsync(stream, ptr), "attach unified memory to stream")
#define cuda_free(ptr)                                  \
    cuda_check(cudaFree(ptr), "memory deallocation")
#define cuda_memcpy_to_device(dest, src, size)                          \
    cuda_check(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice), "copy to device")
#define cuda_memcpy_from_device(dest, src, size)                        \
    cuda_check(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost), "copy from device")
#define cuda_memcpy_on_device(dest, src, size)                        \
    cuda_check(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice), "copy on device")
#define cuda_memset(dest, val, size)                        \
    cuda_check(cudaMemset(dest, val, size), "memset on device")
#define cuda_device_synchronize() \
    cuda_check(cudaDeviceSynchronize(), "device synchronize")

