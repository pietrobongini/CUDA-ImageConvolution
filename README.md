# ImageConvolution
Implementations of parallel 2D Image Convolution algorithm with CUDA (using global memory, shared memory and constant memory) and C++11

cudaGlobalMemoryConvolution ---> using global memory of GPU
cudaConstantMemoryConvolution ---> using global memory and the mask in constant memory
cudaSharedMemoryConvolution ---> using shared memory of GPU (tiling)
cudaConstantSharedMemoryConvolution ---> using shared memory and the mask in constant memory (tiling)

* chrono library is used to measure the execution time
