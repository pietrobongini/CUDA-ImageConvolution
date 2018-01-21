//
// Created by pietro bongini on 28/09/17.
//

#include "PPM.h"
#include "Utils.h"
#include "Image.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

using namespace std;

#define PPMREADBUFLEN 256

static const char *skipSpaces(const char *line) {
    while (*line == ' ' || *line == '\t') {
        line++;
        if (*line == '\0') {
            break;
        }
    }
    return line;
}

static char nextNonSpaceChar(const char *line0) {
    const char *line = skipSpaces(line0);
    return *line;
}

static bool isComment(const char *line) {
    char nextChar = nextNonSpaceChar(line);
    if (nextChar == '\0') {
        return true;
    } else {
        return nextChar == '#';
    }
}

static void parseDimensions(const char *line0, int *width, int *height) {
    const char *line = skipSpaces(line0);
    sscanf(line, "%d %d", width, height);
}

static void parseDimensions(const char *line0, int *width, int *height,
                            int *channels) {
    const char *line = skipSpaces(line0);
    sscanf(line, "%d %d %d", width, height, channels);
}

static void parseDepth(const char *line0, int *depth) {
    const char *line = skipSpaces(line0);
    sscanf(line, "%d", depth);
}

static char *File_readLine(FILE* file) {
    static char buffer[PPMREADBUFLEN];
    if (file == NULL) {
        return NULL;
    }
    memset(buffer, 0, PPMREADBUFLEN);

    if (fgets(buffer, PPMREADBUFLEN - 1, file)) {
        return buffer;
    } else {
        return NULL;
    }
}

static char *nextLine(FILE* file) {
    char *line = NULL;
    while ((line = File_readLine(file)) != NULL) {
        if (!isComment(line)) {
            break;
        }
    }
    return line;
}

char* File_read(FILE* file, size_t size, size_t count) {
    size_t res;
    char *buffer;
    size_t bufferLen;

    if (file == NULL) {
        return NULL;
    }

    bufferLen = size * count + 1;
    buffer = (char*) malloc(sizeof(char) * bufferLen);

    res = fread(buffer, size, count, file);
    // make valid C string
    buffer[size * res] = '\0';

    return buffer;
}

bool File_write(FILE* file, const void *buffer, size_t size, size_t count) {
    if (file == NULL) {
        return false;
    }

    size_t res = fwrite(buffer, size, count, file);
    if (res != count) {
        printf("ERROR: Failed to write data to PPM file");
    }

    return true;
}

Image_t* PPM_import(const char *filename) {
    Image_t* img;
    FILE* file;
    char *header;
    char *line;
    int ii, jj, kk, channels;
    int width, height, depth;
    unsigned char *charData, *charIter;
    float *imgData, *floatIter;
    float scale;

    img = NULL;

    file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Could not open %s\n", filename);
        goto cleanup;
    }

    header = File_readLine(file);
    if (header == NULL) {
        printf("Could not read from %s\n", filename);
        goto cleanup;
    } else if (strcmp(header, "P6") != 0 && strcmp(header, "P6\n") != 0
               && strcmp(header, "P5") != 0 && strcmp(header, "P5\n") != 0
               && strcmp(header, "S6") != 0 && strcmp(header, "S6\n") != 0) {
        printf("Could not find magic number for %s\n", filename);
        goto cleanup;
    }

    // P5 are monochrome while P6/S6 are RGB
    // S6 needs to parse number of channels out of file
    if (strcmp(header, "P5") == 0 || strcmp(header, "P5\n") == 0) {
        channels = 1;
        line = nextLine(file);
        parseDimensions(line, &width, &height);
    } else if (strcmp(header, "P6") == 0 || strcmp(header, "P6\n") == 0) {
        channels = 3;
        line = nextLine(file);
        parseDimensions(line, &width, &height);
    } else {
        line = nextLine(file);
        parseDimensions(line, &width, &height, &channels);
    }

    // the line now contains the depth information
    line = nextLine(file);
    parseDepth(line, &depth);

    // the rest of the lines contain the data in binary format
    charData = (unsigned char *) File_read(file,
                                           width * channels * sizeof(unsigned char), height);

    img = Image_new(width, height, channels);

    imgData = Image_getData(img);

    charIter = charData;
    floatIter = imgData;
    scale = 1.0f / ((float) depth);

    for (ii = 0; ii < height; ii++) {
        for (jj = 0; jj < width; jj++) {
            for (kk = 0; kk < channels; kk++) {
                *floatIter = ((float) *charIter) * scale;
                floatIter++;
                charIter++;
            }
        }
    }

    cleanup: fclose(file);
    return img;
}

bool PPM_export(const char *filename, Image_t* img) {
    int ii;
    int jj;
    int kk;
    int depth;
    int width;
    int height;
    int channels;
    FILE* file;
    float *floatIter;
    unsigned char *charData;
    unsigned char *charIter;

    file = fopen(filename, "wb+");
    if (file == NULL) {
        printf("Could not open %s in mode %s\n", filename, "wb+");
        return false;
    }

    width = Image_getWidth(img);
    height = Image_getHeight(img);
    channels = Image_getChannels(img);
    depth = 255;

    if (channels == 1) {
        fprintf(file, "P5\n");
    } else {
        fprintf(file, "P6\n");
    }
    fprintf(file, "#Created via PPM Export\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "%d\n", depth);

    charData = (unsigned char*) malloc(
            sizeof(unsigned char) * width * height * channels);

    charIter = charData;
    floatIter = Image_getData(img);
    for (ii = 0; ii < height; ii++) {
        for (jj = 0; jj < width; jj++) {
            for (kk = 0; kk < channels; kk++) {
                *charIter = (unsigned char) ceil(
                        clamp(*floatIter, 0, 1) * depth);
                floatIter++;
                charIter++;
            }
        }
    }

    bool writeResult = File_write(file, charData,
                                  width * channels * sizeof(unsigned char), height);

    free(charData);
    fflush(file);
    fclose(file);

    return true;
}
