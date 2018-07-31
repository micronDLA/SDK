#ifndef _SNOWFLAKE_API_H_INCLUDED_
#define _SNOWFLAKE_API_H_INCLUDED_

#ifdef __cplusplus
extern "C" {
#endif

void *snowflake_compile(const char *test, const char *param, const char *image, const char *modeldir, const char* outbin,
                     unsigned *swoutsize, int numcard, int numclus, int nlayers);

void *snowflake_init(void *cmemo, const char* fbitfile, const char* inbin, unsigned* outsize);

int snowflake_setflag(const char *name, const char *value);

int snowflake_getinfo(const char *name, void *value);

int snowflake_run(void *cmemo, const float *input, unsigned input_elements, float *output, unsigned output_elements);

int snowflake_run_sim(void *cmemo, const float *input, unsigned input_elements, float *output, unsigned output_elements);//only runs the SF_INT precision in software (naive)

int thnets_run_sim(void *cmemo, const float *input, unsigned input_elements, float *output, unsigned output_elements);

int test_functions(void *cmemo, const float *input, unsigned input_elements, float *output, unsigned output_elements);

void snowflake_free(void* cmemo);

#ifdef __cplusplus
}
#endif

#endif
