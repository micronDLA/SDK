/*
Example to run FWDNXT inference engine using put and get_result
*/
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dirent.h>
#include "../../api.h"
//#define STB_IMAGE_IMPLEMENTATION
#include "../../stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../stb_image_resize.h"

static void print_help()
{
    printf("Syntax: simpledemo\n");
    printf("\t-i <directory with image files>\n");
    printf("\t-c <categories file>\t-b <bitfile>\t-s <fwdnxt.bin file>\n");
}

#define BYTE2FLOAT 0.003921568f // 1/255

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
uint64_t outsize = 0;
void *sf_handle;
const char *categ = "./categories.txt";

struct info
{
    float *input;
    char *filename;
};

int sortcmp(const void *a, const void *b)
{
    float diff = sortdata[*(int *)b] - sortdata[*(int *)a];
    if (diff > 0)
        return 1;
    else if (diff < 0)
        return -1;
    return 0;
}

void *getresults_thread(void *dummy);

int main(int argc, char **argv)
{
    const char *imagesdir = "images";
    const char *f_bitfile = "";
    const char *outbin = "save.bin";
    int i, netwidth = 224, netheight = 224;
    pthread_t tid;

    // start argc ------------------------
    for(i = 1; i < argc; i++) {
        if(argv[i][0] != '-')
            continue;
        switch(argv[i][1])
        {
        case 'b': // bitfile
            if(i+1 < argc)
                f_bitfile = argv[++i];
            break;
        case 'i':// imagesdir
            if(i+1 < argc)
                imagesdir = argv[++i];
            break;
        case 'r':// resolution WxH
            if(i+1 < argc)
                sscanf(argv[++i], "%dx%d", &netwidth, &netheight);
            break;
        case 'c':// categories
            if(i+1 < argc)
                categ = argv[++i];
            break;
        case 's':// output file
            if(i+1 < argc)
                outbin = argv[++i];
            break;
        default:
            print_help();
            return -1;
        }
    }
    if(argc==1)
    {
        print_help();
        return -1;
    }

    int noutputs;
    sf_handle = ie_init(NULL, f_bitfile, outbin, &outsize, &noutputs);
    pthread_create(&tid, 0, getresults_thread, 0);
    DIR *dir = opendir(imagesdir);
    if (!dir)
    {
        fprintf(stderr, "Cannot open directory %s\n", imagesdir);
        return -1;
    }
    struct dirent *de;
    while ( (de = readdir(dir)) )
    {
        char path[257];
        if (de->d_type != DT_REG)
            continue;
        sprintf(path, "%s/%s", imagesdir, de->d_name);
        float mean[3] = {0.485, 0.456, 0.406};
        float std[3] = {0.229, 0.224, 0.225};
        int width, height, cp;
        unsigned char *bitmap = (unsigned char *)stbi_load(path, &width, &height, &cp, 3);
        if(!bitmap)
        {
            fprintf(stderr, "The image %s could not be loaded\n", path);
            continue;
        }
        unsigned char *resized = (unsigned char *)malloc(3 * netwidth * netheight);
        stbir_resize_uint8(bitmap, width, height, 0, resized, netwidth, netheight,  0, 3);
        free(bitmap);
        float *input = (float *)malloc(sizeof(float) * 3 * netwidth * netheight);
        rgb2float_cmajor(input, resized, netwidth, netheight, 3, netwidth * 3, mean, std);
        uint64_t input_elements = netwidth * netheight * 3;
        free(resized);
        struct info *info = (struct info *)malloc(sizeof(struct info));
        info->input = input;
        info->filename = strdup(de->d_name);
        int err = ie_putinput(sf_handle, (const float * const *)&input, &input_elements, info);
        if(err==-1)
        {
            fprintf(stderr,"Sorry an error occured, please contact fwdnxt for help. We will try to solve it asap\n");
            return -1;
        }
    }
    // Notify we finished
    ie_putinput(sf_handle, 0, 0, 0);
    closedir(dir);
    pthread_join(tid, 0);
    ie_free(sf_handle);
    printf("\ndone\n");
    return 0;
}


void *getresults_thread(void *dummy)
{
    int i;
    uint64_t output_elements = outsize;
    char **categories = (char **)calloc(output_elements, sizeof(char *));
    FILE *fp = fopen(categ, "r");
    if(fp)
    {
        char line[300];
        int i = 0;
        while (i < output_elements && fgets(line, sizeof(line), fp))
        {
            char *p = strchr(line, '\n');
            if(p)
                *p = 0;
            categories[i++] = strdup(line);
        }
        fclose(fp);
    }
    float *output = (float*) malloc(output_elements*sizeof(float));
    for (;;)
    {
        struct info *info;
        int err = ie_getresult(sf_handle, &output, &output_elements, (void **)&info);
        if(err==-1)
        {
            fprintf(stderr,"Sorry an error occured, please contact fwdnxt for help. We will try to solve it asap\n");
            exit(-1);
        }
        if (!info) // We sent an empty input to notify that we finished
            break;
        printf("-------------- %s --------------\n", info->filename);
        int* idxs = (int *)malloc(sizeof(int) * output_elements);
        for(i = 0; i < output_elements; i++)
            idxs[i] = i;
        sortdata = output;
        qsort(idxs, outsize, sizeof(int), sortcmp);
        for(i = 0; i < 5; i++)
            printf("%s (%d) -- %.4f\n", categories[idxs[i]] ? categories[idxs[i]] : "", idxs[i], output[idxs[i]]);
        free(idxs);
        free(info->input);
        free(info->filename);
        free(info);
    }
    free(output);
    for(i = 0; i < output_elements; i++)
        if(categories[i])
            free(categories[i]);
    free(categories);
}
