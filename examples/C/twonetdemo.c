/*
Author: Andre Chang
Run FWDNXT inference engine instructions
*/
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "../../api.h"
//#define STB_IMAGE_IMPLEMENTATION
#include "../../stb_image.h"

static void print_help()
{
    printf("Syntax: twonetdemo <bin1> <bin2> <image1> <image2>\n");
    printf("\t-c <categories file>\n\t-b <bitfile>\n");
    printf("\t-f <number of FPGAs to use>\n\t-C <number of clusters>\n");
}

#define BYTE2FLOAT 0.003921568f // 1/255

//convert rgb image into float and arrange into column first ordering
void rgb2float_cmajor(float *dst, const unsigned char *src, int width, int height, int cp, int srcstride, const float *mean, const float *std)
{
    int c, i, j;
    float std1[3];
    for(i = 0; i < cp; i++)
        std1[i] = 1 / std[i];
    for(c = 0; c < cp; c++)
        for(i = 0; i < height; i++)
            for(j = 0; j < width; j++)
                dst[c*width*height + i*width + j] = (src[c + cp*j + srcstride*i] * BYTE2FLOAT - mean[c]) * std1[c];
}

float *sortdata;
//sort the vector
int sortcmp(const void *a, const void *b)
{
    float diff = sortdata[*(int *)b] - sortdata[*(int *)a];
    if (diff > 0)
        return 1;
    else if (diff < 0)
        return -1;
    return 0;
}

int main(int argc, char **argv)
{
    const char *image[2] = {0,0};
    const char *categ = "./categories.txt";//categories list
    const char *f_bitfile = "";//FGPA bitfile with FWDNXT inference engine
    const char *binfile[2];
    int nfpga = 1;
    int nclus = 1;
    int i, n = 0;

    //program arguments ------------------------
    for(i = 1; i < argc; i++) {
        if(argv[i][0] != '-')
        {
            if(n < 2)
                binfile[n] = argv[i];
            else if (n < 4)
                image[n-2] = argv[i];
            n++;
            continue;
        }
        switch(argv[i][1])
        {
        case 'f':// number of fpgas
            if(i+1 < argc)
                nfpga = atoi(argv[++i]);
            break;
        case 'C':// number clusters
            if(i+1 < argc)
                nclus = atoi(argv[++i]);
            break;
        case 'c':// categories
            if(i+1 < argc)
                categ = argv[++i];
            break;
        default:
            print_help();
            return -1;
        }
    }
    if(n < 4)
    {
        print_help();
        return -1;
    }
// initialize FPGA: load hardware and load instructions into memory
    printf("Initialize FWDNXT inference engine FPGA\n");
    uint64_t outsize[2];//number of output values produced
    void *sf_handle = ie_loadmulti(0, binfile, 2);
    int noutputs;
    ie_init(sf_handle, f_bitfile, 0, outsize, &noutputs);
    float *input[2] = {NULL, NULL};
    uint64_t input_elements[2] = {0,0};
//fetch input image
    for(n = 0; n < 2; n++)
    {
        float mean[3] = {0.485, 0.456, 0.406};
        float std[3] = {0.229, 0.224, 0.225};
        int width, height, cp;
        unsigned char *bitmap = (unsigned char *)stbi_load(image[n], &width, &height, &cp, 0);
        if(!bitmap)
        {
            printf("The image %s could not be loaded\n", image[n]);
            return -1;
        }
        input[n] = (float *)malloc(sizeof(float) * cp * width * height * 2);
        rgb2float_cmajor(input[n], bitmap, width, height, cp, width * cp, mean, std);
        input_elements[n] = width * height * cp * nfpga * nclus;
        free(bitmap);
    }
    uint64_t output_elements[2] = {outsize[0] * nfpga * nclus, outsize[1] * nfpga * nclus};
    float *output[2];
    output[0] = (float*) malloc(output_elements[0]*sizeof(float));//allocate memory to hold output
    output[1] = (float*) malloc(output_elements[1]*sizeof(float));//allocate memory to hold output
    int err = 0;
// run inference
    printf("Run FWDNXT inference engine\n");
    err = ie_run(sf_handle, (const float * const *)input, input_elements, output, output_elements);
    if(err==-1)
    {
        fprintf(stderr,"Sorry an error occured, please contact fwdnxt for help. We will try to solve it asap\n");
        return -1;
    }
    if(input[0])
        free(input[0]);
    if(input[1])
        free(input[1]);
//read categories list
    FILE *fp = fopen(categ, "r");
    char **categories = (char **)calloc(output_elements[0], sizeof(char *));
    if(fp)
    {
        char line[300];
        int i = 0;
        while (i < output_elements[0] && fgets(line, sizeof(line), fp))
        {
            char *p = strchr(line, '\n');
            if(p)
                *p = 0;
            categories[i++] = strdup(line);
        }
        fclose(fp);
    }
//print out the results
    printf("-------------- Results --------------\n");
    for(int n = 0; n < 2; n++)
    {
        printf("%s\n", binfile[n]);
        int* idxs = (int *)malloc(sizeof(int) * output_elements[n]);
        for(i = 0; i < output_elements[n]; i++)
            idxs[i] = i;
        sortdata = output[n];
        qsort(idxs, output_elements[n], sizeof(int), sortcmp);
        for(i = 0; i < 5; i++)
            printf("%s (%d) -- %.4f\n", categories[idxs[i]] ? categories[idxs[i]] : "", idxs[i], output[n][idxs[i]]);
    //free allocated memory
        free(idxs);
    }
    for(i = 0; i < output_elements[0]; i++)
        if(categories[i])
            free(categories[i]);
    free(categories);
    ie_free(sf_handle);
    if(output[0])
        free(output[0]);
    if(output[1])
        free(output[1]);
    printf("\ndone\n");
    return 0;
}

