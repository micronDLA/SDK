/*
Author:Andre Chang
Compile a model onnx file and generate snowflake instructions
*/
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../api.h"

static void print_help(){
    printf("Syntax: compile\n");
    printf("\t-m <model file>\n\t-i <input image sizes>\n\t-o <output file>\n");
    printf("\t-f <number of FPGAs to use>\n\t-C <number of clusters to use>\n");
}

int main(int argc, char **argv)
{
    const char *modelpath = "./alexnet.onnx";
    const char *image = "224x224x3";//Width x Height x Channels
    const char *outbin = "save.bin";
    int nfpga = 1;
    int nclus = 1;
    int i;
    // arguments of this program ------------------------
    for(i = 1; i < argc; i++) {
        if(argv[i][0] != '-')
            continue;
        switch(argv[i][1])
        {
        case 'f':// number of fpgas
            if(i+1 < argc){
                nfpga = atoi(argv[++i]);
            }
            break;
        case 'm':// modelpath
            if(i+1 < argc){
                modelpath = argv[++i];
            }
            break;
        case 'i':// modelpath
            if(i+1 < argc){
                image = argv[++i];
            }
            break;
        case 'C':// number clusters
            if(i+1 < argc){
                nclus = atoi(argv[++i]);
            }
            break;
        case 'o':// output file
            if(i+1 < argc){
                outbin = argv[++i];
            }
            break;
        default:
            print_help();
            return -1;
            break;
        }
    }
    if(argc==1){
        print_help();
        return -1;
    }
    uint64_t outsize;
    void* sf_handle = snowflake_compile(image, modelpath, outbin, &outsize, nfpga, nclus, -1);
    snowflake_free(sf_handle);
    printf("\ndone\n");
    return 0;
}

