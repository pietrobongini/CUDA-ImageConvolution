
//
// Created by pietro bongini on 28/09/17.
//

#include "Image.h"
#include "Utils.h"
#include <iostream>
#include <cassert>
#include <cstdlib>


//metodo che restituisce l'immagine
Image_t* Image_new(int width, int height, int channels, float *data) {
    Image_t* img;

    img = (Image_t*) malloc(sizeof(Image_t));

    Image_setWidth(img, width);
    Image_setHeight(img, height);
    Image_setChannels(img, channels);
    Image_setPitch(img, width * channels);

    Image_setData(img, data);
    return img;
}

Image_t* Image_new(int width, int height, int channels) {
    float *data = (float*) malloc(sizeof(float) * width * height * channels);
    return Image_new(width, height, channels, data);
}

Image_t* Image_new(int width, int height) {
    return Image_new(width, height, Image_channels);
}

//metodo che cancella l'immagine
void Image_delete(Image_t* img) {
    if (img != NULL) {
        if (Image_getData(img) != NULL) {
            free(Image_getData(img));
        }
        free(img);
    }
}

//metodo setter per un pixel
void Image_setPixel(Image_t* img, int x, int y, int c, float val) {
    float *data = Image_getData(img);
    int channels = Image_getChannels(img);
    int pitch = Image_getPitch(img);

    data[y * pitch + x * channels + c] = val;

    return;
}

//metodo getter per un pixel
float Image_getPixel(Image_t* img, int x, int y, int c) {
    float *data = Image_getData(img);
    int channels = Image_getChannels(img);
    int pitch = Image_getPitch(img);

    return data[y * pitch + x * channels + c];
}

//confronto tra immagini
bool Image_is_same(Image_t* a, Image_t* b) {
    if (a == NULL || b == NULL) {
        std::cerr << "Comparing null images." << std::endl;
        return false;
    } else if (a == b) {
        return true;
    } else if (Image_getWidth(a) != Image_getWidth(b)) {
        std::cerr << "Image widths do not match." << std::endl;
        return false;
    } else if (Image_getHeight(a) != Image_getHeight(b)) {
        std::cerr << "Image heights do not match." << std::endl;
        return false;
    } else if (Image_getChannels(a) != Image_getChannels(b)) {
        std::cerr << "Image channels do not match." << std::endl;
        return false;
    } else {
        float *aData, *bData;
        int width, height, channels;
        int ii, jj, kk;

        aData = Image_getData(a);
        bData = Image_getData(b);

        assert(aData != NULL);
        assert(bData != NULL);

        width = Image_getWidth(a);
        height = Image_getHeight(a);
        channels = Image_getChannels(a);

        for (ii = 0; ii < height; ii++) {
            for (jj = 0; jj < width; jj++) {
                for (kk = 0; kk < channels; kk++) {
                    float x, y;
                    if (channels <= 3) {
                        x = clamp(*aData++, 0, 1);
                        y = clamp(*bData++, 0, 1);
                    } else {
                        x = *aData++;
                        y = *bData++;
                    }
                    if (almostUnequalFloat(x, y)) {
                        std::cerr
                                << "Image pixels do not match at position ( row = "
                                << ii << ", col = " << jj << ", channel = "
                                << kk << ") expecting a value of " << y
                                << " but got a value of " << x << std::endl;

                        return false;
                    }
                }
            }
        }
        return true;
    }
}

