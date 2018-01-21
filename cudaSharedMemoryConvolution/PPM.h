//
// Created by pietro bongini on 28/09/17.
//

#ifndef KERNELPROCESSING_PPM_H
#define KERNELPROCESSING_PPM_H

#include "Image.h"

Image_t* PPM_import(const char *filename);
bool PPM_export(const char *filename, Image_t* img);



#endif //KERNELPROCESSING_PPM_H

