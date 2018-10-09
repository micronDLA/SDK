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
    printf("\t-f <number of FPGAs to use>\n\t-C <number of clusters>\n");
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
int nclus = 1, nfpga = 1;

struct info
{
    float *input;
    char *filename[4];
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
    DIR *dir;
    struct dirent *de;
    struct info *info;
    int batchidx = 0;

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
        case 'f':// number of fpgas
            if(i+1 < argc)
                nfpga = atoi(argv[++i]);
            break;
        case 'i':// imagesdir
            if(i+1 < argc)
                imagesdir = argv[++i];
            break;
        case 'r':// resolution WxH
            if(i+1 < argc)
                sscanf(argv[++i], "%dx%d", &netwidth, &netheight);
            break;
        case 'C':// number clusters
            if(i+1 < argc)
                nclus = atoi(argv[++i]);
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

    sf_handle = ie_init(NULL, f_bitfile, outbin, &outsize);
    pthread_create(&tid, 0, getresults_thread, 0);
    dir = opendir(imagesdir);
    if (!dir)
    {
        fprintf(stderr, "Cannot open directory %s\n", imagesdir);
        return -1;
    }
    while ( (de = readdir(dir)) )
    {
        char path[256];
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
        if(!batchidx)
        {
            info = (struct info *)calloc(1, sizeof(struct info));
            info->input = (float *)calloc(1, sizeof(float) * 3 * netwidth * netheight * nclus * nfpga);
        }
        rgb2float_cmajor(info->input + 3 * netwidth * netheight * batchidx, resized, netwidth, netheight, 3, netwidth * 3, mean, std);
        info->filename[batchidx] = strdup(de->d_name);
        free(resized);
        batchidx++;
        if(batchidx == nclus * nfpga)
        {
            int err = ie_putinput(sf_handle, info->input, netwidth * netheight * 3 * nclus * nfpga, info);
            if(err==-1)
            {
                fprintf(stderr,"Sorry an error occured, please contact fwdnxt for help. We will try to solve it asap\n");
                return -1;
            }
            batchidx = 0;
        }
    }
    if(batchidx)
    {
        // Process what left
        int err = ie_putinput(sf_handle, info->input, netwidth * netheight * 3 * nclus * nfpga, info);
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
    char **categories = (char **)calloc(outsize, sizeof(char *));
    FILE *fp = fopen(categ, "r");
    if(fp)
    {
        char line[300];
        int i = 0;
        while (i < outsize && fgets(line, sizeof(line), fp))
        {
            char *p = strchr(line, '\n');
            if(p)
                *p = 0;
            categories[i++] = strdup(line);
        }
        fclose(fp);
    }
    float *output = (float*) malloc(outsize * sizeof(float) * nclus * nfpga);
    for (;;)
    {
        struct info *info;
        int err = ie_getresult(sf_handle, output, outsize * nclus * nfpga, (void **)&info);
        if(err==-1)
        {
            fprintf(stderr,"Sorry an error occured, please contact fwdnxt for help. We will try to solve it asap\n");
            exit(-1);
        }
        if (!info) // We sent no info to notify that we finished
            break;
        for(int batchidx = 0; batchidx < nclus * nfpga && info->filename[batchidx]; batchidx++)
        {
            printf("-------------- %s --------------\n", info->filename[batchidx]);
            int* idxs = (int *)malloc(sizeof(int) * outsize);
            for(i = 0; i < outsize; i++)
                idxs[i] = i;
            sortdata = output + batchidx * outsize;
            qsort(idxs, outsize, sizeof(int), sortcmp);
            for(i = 0; i < 5; i++)
                printf("%s (%d) -- %.4f\n", categories[idxs[i]] ? categories[idxs[i]] : "", idxs[i], output[idxs[i]]);
            free(idxs);
            free(info->filename[batchidx]);
        }
        free(info->input);
        free(info);
    }
    free(output);
    for(i = 0; i < outsize; i++)
        if(categories[i])
            free(categories[i]);
    free(categories);
}
