/*
 *	Returns the abbreviated property's list for all CUDA devices.
 */

#include <stdio.h>

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("\nnvcc (CUDA toolkit) version: %d.%d.%d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);

    int driverVersion;
    if (cudaDriverGetVersion(&driverVersion) == cudaSuccess)
        printf("CUDA driver version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    else
        printf("CUDA driver version: NA\n");

    int runtimeVersion;
    if (cudaRuntimeGetVersion(&runtimeVersion) == cudaSuccess)
        printf("CUDA runtime version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
    else
        printf("CUDA runtime version: NA\n");


    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("\nDevice Number: %d\n", i);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Device name: %s\n", deviceProp.name);
        printf("  Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8.0) / 1.0e6);
        printf("  Total amount of shared memory per block: %lu\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    }
    printf("\n");
}