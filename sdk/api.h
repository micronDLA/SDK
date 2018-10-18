#ifndef _IE_API_H_INCLUDED_
#define _IE_API_H_INCLUDED_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

void *ie_compile(const char *image, const char *modeldir, const char* outbin,
                     uint64_t *swoutsize, int numcard, int numclus, int nlayers);

void *ie_init(void *cmemo, const char* fbitfile, const char* inbin, uint64_t* outsize);

int ie_setflag(const char *name, const char *value);

int ie_getinfo(void *cmemo, const char *name, void *value);

int ie_run(void *cmemo, const float *input, uint64_t input_elements, float *output, uint64_t output_elements);

int ie_run_sim(void *cmemo, const float *input, uint64_t input_elements, float *output, uint64_t output_elements);

int ie_putinput(void *cmemo, const float *input, uint64_t input_elements, void *userparam);

int ie_getresult(void *cmemo, float *output, uint64_t output_elements, void **userparam);

int thnets_run_sim(void *cmemo, const float *input, unsigned input_elements, float *output, unsigned output_elements);

int test_functions(void *cmemo, const float *input, unsigned input_elements, float *output, unsigned output_elements);

void ie_free(void* cmemo);

#ifdef __cplusplus
}
#endif

#endif
