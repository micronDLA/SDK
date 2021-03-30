#include <float.h>
#include <stdlib.h>
#include <stdbool.h>
#include <setjmp.h>
#include <map>
#include <string>
#include <vector>
#include "thvector.h"
#ifdef MEMORYDEBUG
#include "memorydebug.h"
#endif

#ifdef OPENCL
#include "CL/opencl.h"
#endif

namespace thnets {

enum therror {
    ERR_OPENFILE = -1,
    ERR_READFILE = -2,
    ERR_NOTIMPLEMENTED = -3,
    ERR_CORRUPTED = -4,
    ERR_WRONGOBJECT = -5
};

enum thtype {
    TYPE_NIL = 0,
    TYPE_NUMBER = 1,
    TYPE_STRING = 2,
    TYPE_TABLE = 3,
    TYPE_TORCH = 4,
    TYPE_BOOLEAN = 5,
    TYPE_FUNCTION = 6,
    LEGACY_TYPE_RECUR_FUNCTION = 7,
    TYPE_RECUR_FUNCTION = 8,
    TYPE_BYTE = 100,
    TYPE_CHAR = 101,
    TYPE_SHORT = 102,
    TYPE_INT = 103,
    TYPE_LONG = 104,
    TYPE_FLOAT = 105,
    TYPE_DOUBLE = 106,
    TYPE_STORAGE = 200,
    TYPE_TENSOR = 201,
    TYPE_NNMODULE = 202
};

struct thobject;
struct threcord;

struct table {
    int idx;
    int nrefs;
    int nelem;
    struct threcord *records;
};

struct nnmodule {
    int idx;
    int nrefs;
    char *name;
    struct table *table;
};

struct storage {
    int idx;
    int nrefs;
    int scalartype;
    long nelem;
    void *data;
};

struct tensor {
    int idx;
    int nrefs;
    int scalartype;
    int ndim;
    long *size;
    long *stride;
    long storageoffset;
    struct storage *storage;
};

struct thobject
{
    int type;
    union {
        double number;
        struct {
            int size;
            char *data;
        } string;
        struct table *table;
        struct storage *storage;
        struct tensor *tensor;
        struct nnmodule *nnmodule;
        int boolean;
    };
};

struct threcord {
    struct thobject name;
    struct thobject value;
};

enum THDATATYPE {
#define THN_DATA_TYPE(_name_, _size_, _onnx_enum_, _nnef_name_, _c_name_) \
    DT_ ## _name_,
#include "thnets.def"
#undef THN_DATA_TYPE
};

typedef struct THNStorage
{
    void *data;
    enum THDATATYPE datatype;
    char datasize;
    long size;
    int nref, mustfree; // mustfree = 0 (allocated somewhere else), 1 (free), 2 (cuda free)
} THNStorage;

#define MAX_DIM 6 // Maximum number of supported dimensions

typedef struct THNTensor
{
    long size[MAX_DIM];
    long stride[MAX_DIM];
    int nDimension;
    THNStorage *storage;
    long storageOffset;
    int bufferid;
    enum THDATATYPE datatype;
    char datasize;
#ifdef LOWP
    float sub, mult;
#endif
} THNTensor;

struct SpatialConvolution
{
    THNTensor *bias, *weight, *finput;
    int dW, dH, dZ, padW, padH, padZ, kW, kH, kZ, nInputPlane, nOutputPlane;
    int refl_pad;
    int padW2, padH2, padZ2; // right and bottom, if different
    int autopad; // ONNX, 0 = VALID, 1 = SAME_UPPER, 2 = SAME_LOWER
    int dlH, dlW, dlZ; // Dilations
};

struct SpatialFullConvolution
{
    THNTensor *bias, *weight;
    int dW, dH, dZ, padW, padH, padZ, kW, kH, kZ, nInputPlane, nOutputPlane;
    int adjW, adjH, adjZ;
    THNTensor *ones, *columns;
};

struct SpatialMaxPooling
{
    int padW, padH, padZ, dW, dH, dZ, kW, kH, kZ, ceil_mode;
    int iwidth, iheight;
    THNTensor *indices;
    int padW2, padH2, padZ2; // right and bottom, if different
    int autopad; // ONNX, 0 = VALID, 1 = SAME_UPPER, 2 = SAME_LOWER
};

struct SpatialAveragePooling
{
    int padW, padH, padZ, dW, dH, dZ, kW, kH, kZ, ceil_mode;
    int count_include_pad;
    int padW2, padH2, padZ2; // right and bottom, if different
    int autopad; // ONNX, 0 = VALID, 1 = SAME_UPPER, 2 = SAME_LOWER
};

struct Linear
{
    THNTensor *bias, *weight, *addBuffer;
    int commute;    // Used for ONNX, if 1, invert A and B
};

struct Threshold
{
    float threshold, val, alpha;
    float min, max;//clip min and max
    int inplace;
};

struct View
{
    int numElements, nDimension;
    long size[MAX_DIM];
    struct { THNTensor *data, *shape; };  // Reshape
};

struct Dropout
{
    float p;
    int inplace, v2;
};

struct SpatialZeroPadding
{
    int pad_l, pad_r, pad_t, pad_b;
};

struct Reshape
{
    int numElements, batchMode;
    long size[MAX_DIM], batchsize[MAX_DIM];
    int nsize, nbatchsize;
};

struct SpatialMaxUnpooling
{
    struct nnmodule *pooling;
};

struct SpatialBatchNormalization
{
    THNTensor *running_mean, *running_var, *weight, *bias;
    double eps;
};

struct Concat
{
    struct network *net;
    int dimension;
};

struct Cast
{
    THDATATYPE to;
};

struct ConstantOfShape
{
    THNTensor *value;
};

struct Gather
{
    int axis;
    THNTensor *data, *indices;
};

struct InstanceNormalization
{
    float epsilon;
    THNTensor *scale, *bias;
};

struct LRN
{
    float alpha, beta, bias;
    struct { size_t nsize; long size[MAX_DIM]; };
};

struct Sequential
{
    struct network *net;
};

struct PReLU
{
    THNTensor *weight;
    int nOutputPlane;
};

struct Padding
{
    float dim, pad, nInputDim, index, value;
};

struct Slice
{
    int axis, from, to;
    struct { size_t naxes; long starts[MAX_DIM], ends[MAX_DIM], axes[MAX_DIM], steps[MAX_DIM]; };
};

struct Upsample
{
    float width_scale, height_scale;
};

struct LSTM
{
    THNTensor *W, *R, *B;
    int activations[3];
};

struct GRU
{
    THNTensor *W, *R, *B;
    int activations[2];
};

struct Squeeze
{
    struct { size_t naxes; int axes[MAX_DIM]; };
};

struct Tile
{
    struct { size_t nrepeats; long repeats[MAX_DIM]; };
};

struct Pad
{
    int npads;
    long pads[MAX_DIM * 2];
};


enum moduletype {
#define THN_MODULE_TYPE(_name_) MT_ ## _name_,
#include "thnets.def"
#undef THN_MODULE_TYPE
};

struct network;

struct module
{
    module()
      : type(MT_UNDEFINED), updateOutput(0), nnfree(0),
        output(0), outputs(), net(0), nnmodule(0),
#ifdef OPENCL
        kernel{}, clstatus(0),
#endif // OPENCL
        ninputs(0), noutputs(0), isoutput(0), inputs{}, outputname{}, inputnames{}
    {
//printf("module::module(): type:%d\n", type);
    }
    ~module();

    moduletype type;
    THNTensor *(*updateOutput)(struct module *m, THNTensor *in);
    void (*nnfree)(struct module *m);
    THNTensor *output;
    THNTensor *outputs[3]; // Only for LSTM and GRU that have multiple outputs
    struct network *net;
    struct nnmodule *nnmodule;
#ifdef OPENCL
    cl_kernel kernel;
    int clstatus;
#endif
    // These are currently used only by ONNX
    // They are always present in order not to require to define ONNX
    // when including this header
    int ninputs;
    int noutputs;
    char isoutput;
#define MAXMODULEINPUTS 16
    int inputs[MAXMODULEINPUTS];
    char *outputname[MAXMODULEINPUTS];//outputname: lstm have multiple outputs
    char *inputnames[MAXMODULEINPUTS];
    std::vector<THNTensor*> all_inputs;
    // End ONNX
    union {
        struct SpatialConvolution SpatialConvolution;
        struct SpatialMaxPooling SpatialMaxPooling;
        struct SpatialAveragePooling SpatialAveragePooling;
        struct Linear Linear;
        struct Threshold Threshold;
        struct View View;
        struct Dropout Dropout;
        struct SpatialZeroPadding SpatialZeroPadding;
        struct Reshape Reshape;
        struct SpatialFullConvolution SpatialFullConvolution;
        struct SpatialMaxUnpooling SpatialMaxUnpooling;
        struct SpatialBatchNormalization SpatialBatchNormalization;
        struct Sequential Sequential;
        struct Concat Concat;
        struct Sequential ConcatTable;
        struct Concat JoinTable;
        struct PReLU PReLU;
        struct Slice Slice;
        struct Upsample Upsample;
        struct LSTM LSTM;
        struct GRU GRU;
        struct Squeeze Squeeze;
        struct Tile Tile;
        struct Cast Cast;
        struct ConstantOfShape ConstantOfShape;
        struct Gather Gather;
        struct InstanceNormalization InstanceNormalization;
        struct LRN LRN;
        struct Pad Pad;
    };
};

int getoutput(struct network *net, const std::string& name);


enum th_engine {
    ENGINE_CPU,
    ENGINE_CUDA,
    ENGINE_OPENCL,
    ENGINE_OPENCLINIT,
    ENGINE_LOWP,
    ENGINE_TABLE
};

struct network
{
    network(th_engine engine, int nalloc);
    ~network();

    th_engine engine;
    int nelem, nalloc;
    module *const modules;
    std::string inshapes; // String describing input shapes
    std::vector<thnets::THNTensor *> in_t;// tensors describing inputs shapes

    std::string onnx_domain;                    // (ONNX only) opset domain
    long onnx_version;                          // (ONNX only) opset version

    std::map<std::string,THNTensor> tensors;    // list of tensors in the network
    std::vector<std::string> inputs;            // list of input tensor ids

    // Get network input index. Inputs are encoded as negative numbers,
    // the first input is -1, the second -2, ...
    // Return 0 if name is not an input.
    int getInput(const std::string& name) const
    {
        for (size_t i = 0; i < inputs.size(); ++i)
            if (inputs[i] == name)
                return -1 - i;
        return 0;
    }
};

struct object2module
{
    const char *name;
    int (*func)(struct module *mod, struct nnmodule *n);
};

extern struct object2module object2module[];
extern jmp_buf therror_env;
extern char therror[1000];

double TableGetNumber(struct table *t, const char *name);
int TableGetBoolean(struct table *t, const char *name);
THNTensor *TableGetTensor(struct table *t, const char *name);
void *TableGetStorage(struct table *t, const char *name, int *nelem);
struct nnmodule *TableGetNNModule(struct table *t, const char *name);
[[noreturn]] void THError(const char *fmt, ...);
THNTensor *THNTensor_new(enum THDATATYPE datatype);
THNStorage *THNStorage_new(long size, enum THDATATYPE datatype);
THNStorage *THNStorage_newwithbuffer(void *buffer, enum THDATATYPE datatype);
THNTensor *THNTensor_newWithStorage1d(THNStorage *storage, long storageOffset, long size0, long stride0);
THNTensor *THNTensor_newWithStorage2d(THNStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1);
THNTensor *THNTensor_newWithStorage3d(THNStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1, long size2, long stride2);
THNTensor *THNTensor_newWithTensor(THNTensor *tensor);
bool THNTensor_isZero(THNTensor *t);
void THNTensor_defaultStrides(THNTensor *t);
void THNTensor_transpose(THNTensor *tdst, THNTensor *tsrc, int dimension1, int dimension2);
THNTensor *THNTensor_newTranspose(THNTensor *tensor, int dimension1_, int dimension2_);
void *THNTensor_data(THNTensor *tensor);
float *THNTensor_fdata(THNTensor *tensor);
long *THNTensor_ldata(THNTensor *tensor);
int THNTensor_isSameSizeAs(const THNTensor *self, const THNTensor* src);
const std::string THNTensor_Shape(THNTensor *t);
void THNTensor_resize(THNTensor *t, long *size, int nDimension);
void THNTensor_resizeNoStorage(THNTensor *t, long *size, int nDimension);
void THNTensor_resize4d(THNTensor *t, long size0, long size1, long size2, long size3);
void THNTensor_resize3d(THNTensor *t, long size0, long size1, long size2);
void THNTensor_resize2d(THNTensor *t, long size0, long size1);
void THNTensor_resize1d(THNTensor *t, long size0);
void THNTensor_resizeAs(THNTensor *tdst, THNTensor *tsrc);
long THNTensor_nElement(THNTensor *t);
void THNTensor_set(THNTensor *tdst, THNTensor *tsrc);
void THNTensor_zero(THNTensor *t);
void THNTensor_fill(THNTensor *t, float value);
void THNTensor_copy(THNTensor *tdst, THNTensor *tsrc);
void THNTensor_safecopy(THNTensor *tdst, THNTensor *tsrc);
void THNTensor_slice(THNTensor *dst, THNTensor *src, int dimension, long from, long to);
void THNTensor_free(THNTensor *t);
void THNStorage_free(THNStorage *s);
THNTensor *THNTensor_newSelect(THNTensor *tensor, int dimension, long sliceIndex);
THNTensor *THNTensor_squeeze(THNTensor *t);
double THExpMinusApprox(double x);
void THBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);
void THNTensor_addmm(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *m1, THNTensor *m2);
void THNTensor_convmm(THNTensor *r, float beta, float alpha, THNTensor *filt, THNTensor *m,
    int kH, int kW, int dH, int dW, int padH, int padW);
void THNTensor_addr(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *vec1, THNTensor *vec2);
void THNTensor_addmv(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *mat, THNTensor *vec);
void THNTensor_conv2Dmm(THNTensor *r_, float beta, float alpha, THNTensor *t_, THNTensor *k_, long srow, long scol, const char *vf, const char *xc);
void THNTensor_conv2Dmv(THNTensor *r_, float beta, float alpha, THNTensor *t_, THNTensor *k_, long srow, long scol, const char *vf, const char *xc);

#define thfmaxf(a,b) ((a) > (b) ? (a) : (b))
#define thfminf(a,b) ((a) < (b) ? (a) : (b))
#define TH_MODULEIDX(a) ((a) >= 0 ? (a) & 0xffffff : (a)) // modules[].inputs[] contains the moduleidx & module output in the 8 MSBs
#define TH_OUTIDX(a) ((a) >= 0 ? (a) >> 24  : (a))

#define THInf FLT_MAX

#ifdef HAVEFP16
void tofp16(__fp16 *dst, const float *src, size_t len);
void fromfp16(float *dst, const __fp16 *src, size_t len);
#endif

int loadtorch(const char *path, struct thobject *obj, int longsize);
int printobject(struct thobject *obj, int indent);
int freeobject(struct thobject *obj);
THNTensor *forward(struct network *net, THNTensor *in);
THNTensor *THNTensor_newFromObject(struct thobject *obj);
struct network *Module2Network(struct nnmodule *obj);
void printtensor(THNTensor *t);
double th_seconds();

void absorb_bn(struct network *net, int bnidx, struct module *convm);

/* High level API */

typedef struct thnetwork
{
    struct thobject *netobj;
    struct thobject *statobj;
    struct network *net;
    THNTensor *out;
    float mean[3], std[3];
} THNETWORK;

void THInit();
THNETWORK *THLoadNetwork(const char *path);
THNETWORK *THSimplify(THNETWORK *network);
THNTensor *THForward(THNETWORK *net, THNTensor *in);
void THMakeSpatial(THNETWORK *network, int size);
int THProcessFloat(THNETWORK *network, float *data, int batchsize, int width, int height, int nplanes, float **result, int *outwidth, int *outheight);
int THProcessImages(THNETWORK *network, unsigned char **images, int batchsize, int width, int height, int stride, float **result, int *outwidth, int *outheight, int bgr);
int THProcessYUYV(THNETWORK *network, unsigned char *image, int width, int height, float **results, int *outwidth, int *outheight);
THNETWORK *THCreateCudaNetwork(THNETWORK *net);
THNETWORK *THCreateOpenCLNetwork(THNETWORK *net);
THNETWORK *THCreateLowpNetwork(THNETWORK *net, float range);
int THCudaHalfFloat(int enable);
int THOpenCLHalfFloat(int enable);
int THUseSpatialConvolutionMM(THNETWORK *network, int mm_type);
void THFreeNetwork(THNETWORK *network);
int THLastError();
extern int th_debug, th_profile, th_minmax;
extern double th_convtot, th_convflops;

#ifdef CUDNN
#include "cudnn/cudnn_th.h"
#endif

#ifdef OPENCL
#include "opencl/opencl_th.h"
#endif

#ifdef LOWP
#include "lowp/lowp.h"
#endif

#ifdef USEQSML
void init_thnets4qsml_conv(THNETWORK *network);
void transform_mem(struct module newmod, int col, int row, int plane, int outp);
float* transform_mem_input(float* in1, int col, int row, int plane);
#endif

}  // namespace thnets

void thload_Conv2d(struct thnets::module *m, float* weight, float* bias,
        int inp, int outp, int kW, int kH, int pW, int pH, int dW, int dH, int dlW, int dlH, int group);
void thload_TransposedConv2d(struct thnets::module *m, float* weight, float* bias,
        int inp, int outp, int kW, int kH, int pW, int pH, int dW, int dH, int opW, int opH, int group);
void thload_Threshold(struct thnets::module *m);
void thload_Maxpool2d(struct thnets::module *m,
        int kW, int kH, int pW, int pH, int dW, int dH, int dlW, int dlH, bool ceil);
void thload_Avgpool2d(struct thnets::module *m,
        int kW, int kH, int pW, int pH, int dW, int dH, bool ceil);
void thload_Linear(struct thnets::module *m, float* weight, float* bias, int i, int o);
void thload_View(struct thnets::module *m);
void thload_Sigmoid(struct thnets::module *m);
void thload_Tanh(struct thnets::module *m);
void thload_BatchNorm(struct thnets::module *m, float* weight, float* bias, float* run_mean, float* run_var, float eps, int len);
void thload_Add(struct thnets::module *m);
void thload_Sub(struct thnets::module *m);
void thload_Concat(struct thnets::module *m, int dim);
void thload_Upsample(struct thnets::module *m, int w_scale, int h_scale);
