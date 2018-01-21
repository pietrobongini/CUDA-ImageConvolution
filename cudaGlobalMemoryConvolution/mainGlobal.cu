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


#define maskCols 5
#define maskRows 5


__global__ void slowKernelProcessing(float * InputImageData, const float *__restrict__ kernel,
		float* outputImageData, int channels, int width, int height){


	float accum;
	int col = threadIdx.x + blockIdx.x * blockDim.x;   //col index
	int row = threadIdx.y + blockIdx.y * blockDim.y;   //row index
	int maskRowsRadius = maskRows/2;
	int maskColsRadius = maskCols/2;


	for (int k = 0; k < channels; k++){      //cycle on kernel channels
		if(row < height && col < width ){
			accum = 0;
			int startRow = row - maskRowsRadius;  //row index shifted by mask radius
			int startCol = col - maskColsRadius;  //col index shifted by mask radius

			for(int i = 0; i < maskRows; i++){ //cycle on mask rows

				for(int j = 0; j < maskCols; j++){ //cycle on mask columns

					int currentRow = startRow + i; // row index to fetch data from input image
					int currentCol = startCol + j; // col index to fetch data from input image

					if(currentRow >= 0 && currentRow < height && currentCol >= 0 && currentCol < width){

							accum += InputImageData[(currentRow * width + currentCol )*channels + k] *
										kernel[i * maskRows + j];
					}
					else accum = 0;
				}

			}
			outputImageData[(row* width + col) * channels + k] = accum;
		}

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


	inputImage = PPM_import("/home/pietrobongini/cuda-workspace/cudaGlobalMemoryConvolution/img/computer_programming.ppm");

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

	dim3 dimGrid(ceil((float) imageWidth/16),
			ceil((float) imageHeight/16));
	dim3 dimBlock(16,16,1);


	cout << "GLOBAL MEMORY KERNEL CONVOLUTION" << endl;
	cout << "image dimensions: "<< imageWidth << "x" << imageHeight << endl;
	cout << "start parallelizing" << endl;
	cout << "elapsed in time: ";
	high_resolution_clock::time_point start= high_resolution_clock::now();

	slowKernelProcessing<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
	imageChannels, imageWidth, imageHeight);

	high_resolution_clock::time_point end= high_resolution_clock::now();
	chrono::duration<double>  duration = end - start;
	cout << duration.count()*1000 << endl;

	cout << "----------------------------------" << endl;

	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight *
			imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

	PPM_export("/home/pietrobongini/cuda-workspace/cudaGlobalMemoryConvolution/output/result.ppm", outputImage);

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

	return 0;


}
