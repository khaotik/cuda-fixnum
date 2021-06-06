#include <stdio.h>
#include <cuda_runtime.h>
int main() {cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf("%d%d",p.major,p.minor);return 0;}
