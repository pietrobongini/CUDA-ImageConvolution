//
// Created by pietro bongini on 28/09/17.
//

#ifndef KERNELPROCESSING_IMAGE_H
#define KERNELPROCESSING_IMAGE_H

#ifndef IMAGE_H_
#define IMAGE_H_

typedef struct {    //struct per l'immagine
    int width;
    int height;
    int channels;
    int pitch;
    float *data;
} Image_t;

#define Image_channels 3

//metodi getter per i vari elementi dell'immagine
#define Image_getWidth(img) ((img)->width)
#define Image_getHeight(img) ((img)->height)
#define Image_getChannels(img) ((img)->channels)
#define Image_getPitch(img) ((img)->pitch)
#define Image_getData(img) ((img)->data)

//metodi setter per i vari elementi dell'immagine
#define Image_setWidth(img, val) (Image_getWidth(img) = val)
#define Image_setHeight(img, val) (Image_getHeight(img) = val)
#define Image_setChannels(img, val) (Image_getChannels(img) = val)
#define Image_setPitch(img, val) (Image_getPitch(img) = val)
#define Image_setData(img, val) (Image_getData(img) = val)

//vari metodi per la creazione dell'immagine
Image_t* Image_new(int width, int height, int channels, float *data);
Image_t* Image_new(int width, int height, int channels);
Image_t* Image_new(int width, int height);

//metodi getter e setter per pixel
float Image_getPixel(Image_t* img, int x, int y, int c);
void Image_setPixel(Image_t* img, int x, int y, int c, float val);

//metodo per cancellare l'immagine
void Image_delete(Image_t* img);

//metodo booleano per confronto immagini
bool Image_is_same(Image_t* a, Image_t* b);

#endif /* IMAGE_H_ */



#endif //KERNELPROCESSING_IMAGE_H
