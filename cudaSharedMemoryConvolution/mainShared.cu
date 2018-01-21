#include "Image.h"
#include "PPM.h"
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>
#include <chrono>
#include <math.h>
using namespace std;
using namespace std:: chrono;


#define TILE_WIDTH 16
#define maskCols 5
#define maskRows 5
#define w (TILE_WIDTH + maskCols -1)


__global__ void tilingKernelProcessing(float * InputImageData, const float *__restrict__ kernel,
		float* outputImageData, int channels, int width, int height){

	__shared__ float N_ds[w][w];  //block of image in shared memory


	// allocation in shared memory of image blocks
	int maskRadius = maskRows/2;
 	for (int k = 0; k <channels; k++) {
 		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
 		int destY = dest/w;     //row of shared memory
 		int destX = dest%w;		//col of shared memory
 		int srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius; // index to fetch data from input image
 		int srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius; // index to fetch data from input image
 		int src = (srcY *width +srcX) * channels + k;   // index of input image
 		if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
 			N_ds[destY][destX] = InputImageData[src];  // copy element of image in shared memory
 		else
 			N_ds[destY][destX] = 0;



 		dest = threadIdx.y * TILE_WIDTH+ threadIdx.x + TILE_WIDTH * TILE_WIDTH;
 		destY = dest/w;
		destX = dest%w;
		srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;
		srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;
		src = (srcY *width +srcX) * channels + k;
		if(destY < w){
			if(srcY>= 0 && srcY < height && srcX>=0 && srcX < width)
				N_ds[destY][destX] = InputImageData[src];
			else
				N_ds[destY][destX] = 0;
		}

 		__syncthreads();


 		//compute kernel convolution
 		float accum = 0;
 		int y, x;
 		for (y= 0; y < maskCols; y++)
 			for(x = 0; x<maskRows; x++)
 				accum += N_ds[threadIdx.y + y][threadIdx.x + x] *kernel[y * maskCols + x];

 		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
 		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
 		if(y < height && x < width)
 			outputImageData[(y * width + x) * channels + k] = accum;
 		__syncthreads();


 	}

}




int main(){

	int imageChannels;
	int imageHeight;
	int imageWidth;
	Image_t* inputImage;
	Image_t* outputImage;
	float* hostInputImageData;
	float* hostOutputImageData;
	float* deviceInputImageData;
	float* deviceOutputImageData;
	float* deviceMaskData;
	float hostMaskData[maskRows * maskCols]={
			0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04,
			0.04, 0.04, 0.04, 0.04, 0.04


	};


	inputImage = PPM_import("/home/pietrobongini/cuda-workspace/cudaSharedMemoryConvolution/img/computer_programming.ppm");

	imageWidth = Image_getWidth(inputImage);
	imageHeight = Image_getHeight(inputImage);
	imageChannels = Image_getChannels(inputImage);

	outputImage = Image_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = Image_getData(inputImage);
	hostOutputImageData = Image_getData(outputImage);

	cudaDeviceReset();
	cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight *
			imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight *
			imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceMaskData, maskRows * maskCols
			* sizeof(float));
	cudaMemcpy(deviceInputImageData, hostInputImageData,
			imageWidth * imageHeight * imageChannels * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData,
				maskRows * maskCols * sizeof(float),
				cudaMemcpyHostToDevice);


	dim3 dimGrid(ceil((float) imageWidth/TILE_WIDTH),
			ceil((float) imageHeight/TILE_WIDTH));
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);


	cout <<"SHARED MEMORY KERNEL CONVOLUTION" << endl;
	cout << "image dimensions: "<< imageWidth << "x" << imageHeight << endl;
	cout << "start parallelizing" << endl;
	cout << "elapsed in time: ";
	high_resolution_clock::time_point start= high_resolution_clock::now();

	tilingKernelProcessing<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
	imageChannels, imageWidth, imageHeight);

	high_resolution_clock::time_point end= high_resolution_clock::now();
	chrono::duration<double>  duration = end - start;
	cout << duration.count()*1000 << endl;
	cout << "----------------------------------" << endl;

	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight *
			imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

	PPM_export("/home/pietrobongini/cuda-workspace/cudaSharedMemoryConvolution/output/result.ppm", outputImage);

	cudaMemset(deviceInputImageData,0,imageWidth * imageHeight *
				imageChannels * sizeof(float));
	cudaMemset(deviceOutputImageData,0,imageWidth * imageHeight *
					imageChannels * sizeof(float));
	cudaMemset(deviceMaskData,0,maskRows * maskCols
				* sizeof(float));
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	Image_delete(outputImage);
	Image_delete(inputImage);


}
