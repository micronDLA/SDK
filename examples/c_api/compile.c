/*
Compile a model onnx file and generate instructions
*/
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../api.h"

static void print_help()
{
    printf("Syntax: compile <model file> [-i <input image sizes>] [-o <output file> (default save.bin)] [-f <number of FPGAs] [-C <number of clusters>]\n");
}

int main(int argc, char **argv)
{
    const char *modelpath = 0;
    const char *inshapes = 0;
    const char *outbin = "save.bin";
    char s[300];
    int nclus = 1, nfpgas = 1;
    int i;
    // arguments of this program ------------------------
    for(i = 1; i < argc; i++)
    {
        if(argv[i][0] != '-')
            modelpath = argv[i];
        else switch(argv[i][1])
        {
        case 'i':// input shapes
            if(i+1 < argc){
                inshapes = argv[++i];
            }
            break;
        case 'o':// output file
            if(i+1 < argc){
                outbin = argv[++i];
            }
            break;
        case 'C':// number clusters
            if(i+1 < argc)
                nclus = atoi(argv[++i]);
            break;
        case 'f':// categories
            if(i+1 < argc)
                nfpgas = atoi(argv[++i]);
            break;
        default:
            print_help();
            return -1;
            break;
        }
    }
    if(!modelpath)
    {
        print_help();
        return -1;
    }
    unsigned noutputs;
    unsigned *noutdims;
    uint64_t **outshapes;
    
    void* sf_handle = ie_safecreate();
    sprintf(s, "%d", nclus);
    ie_setflag(sf_handle, "nclusters", s);
    sprintf(s, "%d", nfpgas);
    ie_setflag(sf_handle, "nfpgas", s);
    ie_compile(sf_handle, modelpath, outbin, inshapes, &noutputs, &noutdims, &outshapes);
    ie_free(sf_handle);
    printf("done\n");
    return 0;
}

