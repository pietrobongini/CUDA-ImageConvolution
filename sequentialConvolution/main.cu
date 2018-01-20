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

void sequentialConvolution(Image_t*inputImage,const float * kernel ,float * outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels)
{
    int i, j, m, n, mm, nn;
    int kCenterX, kCenterY;                         // center index of kernel
    float sum;                                      // accumulation variable
    int rowIndex, colIndex;                         // indice di riga e di colonna

    float * inputImageData = Image_getData(inputImage);
    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;

    for (int k=0; k<channels; k++) {					//cycle on channels
        for (i = 0; i < dataSizeY; ++i)                //cycle on image rows
        {
            for (j = 0; j < dataSizeX; ++j)            //cycle on image columns
            {
                sum = 0;
                for (m = 0; m < kernelSizeY; ++m)      //cycle kernel rows
                {
                    mm = kernelSizeY - 1 - m;       // row index of flipped kernel

                    for (n = 0; n < kernelSizeX; ++n)  //cycle on kernel columns
                    {
                        nn = kernelSizeX - 1 - n;   // column index of flipped kernel

                        // indexes used for checking boundary
                        rowIndex = i + m - kCenterY;
                        colIndex = j + n - kCenterX;

                        // ignore pixels which are out of bound
                        if (rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
                            sum += inputImageData[(dataSizeX * rowIndex + colIndex)*channels + k] * kernel[kernelSizeX * mm + nn];
                    }
                }
                outputImageData[(dataSizeX * i + j)*channels + k] = sum;

            }
        }
    }
}

int main() {

    Image_t *inputImage;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    float *InputImageData;
    Image_t *outputImage;

    const int kernelRows = 5;
    const int kernelColumns = 5;
    const float kernelMask[kernelRows * kernelColumns] = {0.04, 0.04, 0.04, 0.04, 0.04,
                                                          0.04, 0.04, 0.04, 0.04, 0.04,
                                                          0.04, 0.04, 0.04, 0.04, 0.04,
                                                          0.04, 0.04, 0.04, 0.04, 0.04,
                                                          0.04, 0.04, 0.04, 0.04, 0.04,



    };

    inputImage = PPM_import("/home/pietrobongini/cuda-workspace/sequentialConvolution/img/computer_programming.ppm");
    imageWidth = Image_getWidth(inputImage);
    imageHeight = Image_getHeight(inputImage);
    imageChannels = Image_getChannels(inputImage);
    cout << "dimensioni immagine: " << imageWidth << " x " << imageHeight << endl;
    cout << "numero canali: " << imageChannels << endl;
    InputImageData = Image_getData(inputImage);
    float *outputImageData = Image_getData(inputImage);

    outputImage = inputImage;
    cout << "-------------------------" << endl;
	cout << "sequential kernel convolution" <<endl;
	cout << "start kernel processing... " << endl;
	high_resolution_clock::time_point startSeq = high_resolution_clock::now();
	sequentialConvolution(inputImage, kernelMask, outputImageData, kernelColumns, kernelRows, imageWidth,
						  imageHeight, imageChannels);
	high_resolution_clock::time_point endSeq = high_resolution_clock::now();
	auto durationSeq = (double) duration_cast<milliseconds>(endSeq - startSeq).count() / 1000;
	cout << "end kernel processing" << endl;
	cout << "elapsed in time: " << durationSeq << endl;
	Image_setData(outputImage, outputImageData);
	PPM_export("/home/pietrobongini/cuda-workspace/sequentialConvolution/output/result.ppm", outputImage);

	cout << "-------------------------" << endl;

}
