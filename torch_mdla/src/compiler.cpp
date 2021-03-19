#include "compiler.h"
#include <stack>
#include <c10/util/C++17.h>

#include "thnets.h"

using namespace torch::jit;


c10::List<int64_t> get_const_intlist(Value* input){
    Node* nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isIntList());
    return ivalue->toIntList();
}
int64_t get_const_int(Value* input){
    Node* nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isInt());
    return ivalue->toInt();
}
double get_const_double(Value* input){
    Node* nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isDouble());
    return ivalue->toDouble();
}
bool get_const_bool(Value* input){
    Node* nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isBool());
    return ivalue->toBool();
}
at::Tensor get_tensor(IValue* input){
    assert(input->isTensor());
    return input->toTensor();
}
c10::List<at::Tensor> get_listtensor(IValue* input){
    assert(input->isTensorList());
    return input->toTensorList();
}

static std::string inputnames[MAXMODULEINPUTS];
static int ninputnames;

static int get_node_inputnames(Node *node, thnets::network *net, int n){
    std::string in_id = std::to_string(node->input(n)->unique());
    int mod_idx = getoutput(net, in_id.c_str());//index of module that created this input
    if(mod_idx == -1)//there isn't a module that created that identifier. It must be model input
    {
        int k = -1;
        for(int x = 0; x < ninputnames; x++)
            if(inputnames[x] == in_id){
                k = x;
                break;
            }
        if(k == -1) {
            if(ninputnames == MAXMODULEINPUTS)
                thnets::THError("Maximum number of inputs (%d) exceeded\n", MAXMODULEINPUTS);
            else {
                inputnames[ninputnames] = in_id;
                k = ninputnames++;
            }
        }
        mod_idx = -1 - k; // Inputs are numbered -1, -2, -3...
    }
    return mod_idx;
}

static void print_node(Node* node){
    std::cout << "Running this " << node->kind().toDisplayString() << " inputs: " << node->inputs().size() << std::endl;
    std::cout << "input:";
    for(int ii = 0; ii < node->inputs().size(); ii++){
        std::cout << node->inputs()[ii]->unique() << ", ";
    }
    std::cout << std::endl;
    std::cout << "output: ";
    for(int ii = 0; ii < node->outputs().size(); ii++){
        std::cout << node->outputs()[ii]->unique() << ", ";
    }
    std::cout << std::endl;
}


int MDLACompiler::count_output_used(const char *name)
{
	int count = 0;
	for (auto node : subgraph_->nodes()) {
		for(int j = 0; j < node->inputs().size(); j++)
			if(!strcmp(std::to_string(node->input(j)->unique()).c_str(), name))
				count++;
	}
	return count;
}

void* fcmem = 0;//first cmem has the pico objs
uint64_t laddr_off = 0;//address to combine models in memory

void MDLACompiler::run(torch::jit::Stack* stack) {
    // Get the number of expected inputs to the graph we are compiling
    const at::ArrayRef<Value*>& graph_inputs = subgraph_->inputs();
    const auto num_inputs = graph_inputs.size();
    // Pop these inputs from the stack.
    at::ArrayRef<IValue> inputs = last(stack, num_inputs);
    //map from IValue in stack to node input Value in subgraph_->inputs. IValue has data, Value is just pointer
    std::unordered_map<Value*, IValue> value_to_ivalue;
    for (auto i = 0; i < inputs.size(); ++i) {
        auto value_input = subgraph_->inputs()[i];
        value_to_ivalue[value_input] = inputs[i];
    }
    void *cmem;
    CompleteArgumentSpec spec{false, ArrayRef<IValue>(inputs)};
    Compiled_info* cinfo = NULL;
    at::Tensor in_data;
    bool first = false;
    unsigned ninputs = 1;
    //compile if we haven't compiled for the shape/device of these inputs before
    if (cache_.find(spec) == cache_.end()) {
        cmem = ie_create();
        if(!fcmem) {//first cmem has the pico obj to be copied to other cmem handles
            first = true;
        } else { //combine the code in main memory and use same pico obj
            char s[100];
            sprintf(s, "%ld", laddr_off);
            ie_setflag(cmem, "addr_off", s);
        }

        //create net
        thnets::THInit();
        thnets::THNETWORK *net;
        net = (thnets::THNETWORK *) calloc(1, sizeof(*net));
        net->std[0] = net->std[1] = net->std[2] = 1;
        net->mean[0] = net->mean[1] = net->mean[2] = 0;
        thnets::network* tnet = new thnets::network(thnets::ENGINE_CPU, g_size * 2);
        // Overallocate modules by a factor of 2, because of split and multiple inputs
        tnet->nelem = 0;
        ninputnames = 0;
        int n = 0;

        Node* in_node = NULL;//get nodes that have input
        // Iterating over graph nodes is guaranteed to be topologically sorted
        for (auto node : subgraph_->nodes()) {
            //print_node(node);
            //TODO: add more layers
            //node kind specifies what operation it is. All pytorch kinds are auto generated in: pytorch/torch/csrc/jit/generated/
            if(canHandle(node) && node->kind() != prim::Constant &&  node->kind() != prim::ListConstruct) {
                int num_input = 1;
                if(!in_node)//get first node of graph to get input shape
                    in_node = node;
                if(node->kind() == aten::conv2d || node->kind() == aten::_convolution || node->kind() == aten::conv1d){
                //at::conv2d(input, weight, bias, stride, padding, dilation, groups);
                //at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
                    assert(value_to_ivalue.find(node->input(1)) != value_to_ivalue.end());
                    at::Tensor kk = get_tensor(&value_to_ivalue[node->input(1)]);
                    float* bias = NULL;
                    if(value_to_ivalue.find(node->input(2)) != value_to_ivalue.end()){
                        at::Tensor bb = get_tensor(&value_to_ivalue[node->input(2)]);
                        bias = (float*) bb.data_ptr();
                    }
                    int outp, inp, kW, kH = 1, dW, dH = 1, pW, pH = 0, dlW, dlH = 1, opW = 0, opH = 0, group;
                    bool transpose = false;
                    if(node->kind() == aten::conv1d){//conv1d
                        outp = kk.sizes()[0];
                        inp = kk.sizes()[1];
                        kW = kk.sizes()[2];
                        dW = get_const_intlist(node->input(3))[0];
                        pW = get_const_intlist(node->input(4))[0];
                        dlW = get_const_intlist(node->input(5))[0];
                        group = get_const_int(node->input(6));
                    }
                    else{//conv2d
                        outp = kk.sizes()[0];
                        inp = kk.sizes()[1];
                        kW = kk.sizes()[3];
                        kH = kk.sizes()[2];
                        if(node->kind() == aten::_convolution){
                            dH = dW = get_const_intlist(node->input(3))[0];
                            pH = pW = get_const_intlist(node->input(4))[0];
                            dlH = dlW = get_const_intlist(node->input(5))[0];
                            opH = opW = get_const_intlist(node->input(7))[0];
                            transpose = get_const_bool(node->input(6));
                            group = get_const_int(node->input(8));
                        } else {
                            dW = get_const_intlist(node->input(3))[0];
                            dH = get_const_intlist(node->input(3))[1];
                            pW = get_const_intlist(node->input(4))[0];
                            pH = get_const_intlist(node->input(4))[1];
                            dlW = get_const_intlist(node->input(5))[0];
                            dlH = get_const_intlist(node->input(5))[1];
                            group = get_const_int(node->input(6));
                        }
                    }
                    //printf("spconv_%dx%dx%ds%dx%dp%dx%ddl%dx%dgrp%d", outp, kW, kH, dW, dH, pW, pH, dlW, dlH, group);
                    float* weight = (float*) kk.data_ptr();
                    if(transpose){
                        int tt = inp;//swap inp and outp
                        inp = outp;
                        outp = tt;
                        thload_TransposedConv2d(tnet->modules + n, weight, bias, inp, outp, kW, kH, pW, pH, dW, dH, opW, opH, group);
                    }
                    else
                        thload_Conv2d(tnet->modules + n, weight, bias, inp, outp, kW, kH, pW, pH, dW, dH, dlW, dlH, group);
                }
                else if(node->kind() == aten::linear){
                    assert(value_to_ivalue.find(node->input(1)) != value_to_ivalue.end());
                    at::Tensor kk = get_tensor(&value_to_ivalue[node->input(1)]);
                    float* weight = (float*) kk.data_ptr();
                    int i = 0, o = 0;
                    o = kk.sizes()[0];
                    i = kk.sizes()[1];
                    float* bias = NULL;
                    if(value_to_ivalue.find(node->input(2)) != value_to_ivalue.end()){
                        at::Tensor bb = get_tensor(&value_to_ivalue[node->input(2)]);
                        bias = (float*) bb.data_ptr();
                    }
                    thload_Linear(tnet->modules + n, weight, bias, i, o);
                }
                else if(node->kind() == aten::cat){
                    //c10::List<at::Tensor> lten = get_listtensor(&value_to_ivalue[node->input(0)]);
                    Node* nn = node->input(0)->node();
                    int axis = get_const_int(node->input(1));
                    num_input = nn->inputs().size();
                    for (unsigned x = 0; x < nn->inputs().size(); x++)
                        tnet->modules[n].inputs[tnet->modules[n].ninputs++] = get_node_inputnames(nn, tnet, x);
                    thload_Concat(tnet->modules + n, axis);
                }
                else if(node->kind() == aten::upsample_nearest2d){
                    //int h_scale = get_const_intlist(node->input(1))[0];
                    //int w_scale = get_const_intlist(node->input(1))[1];
                    thload_Upsample(tnet->modules + n, 2, 2);
                }
                else if(node->kind() == aten::add){
                    num_input = 2;
                    tnet->modules[n].inputs[tnet->modules[n].ninputs++] = get_node_inputnames(node, tnet, 0);
                    tnet->modules[n].inputs[tnet->modules[n].ninputs++] = get_node_inputnames(node, tnet, 1);
                    thload_Add(tnet->modules + n);
                }
                else if(node->kind() == aten::sub){
                    num_input = 2;
                    tnet->modules[n].inputs[tnet->modules[n].ninputs++] = get_node_inputnames(node, tnet, 0);
                    tnet->modules[n].inputs[tnet->modules[n].ninputs++] = get_node_inputnames(node, tnet, 1);
                    thload_Sub(tnet->modules + n);
                }
                else if(node->kind() == aten::mul){
                    num_input = 2;
                    tnet->modules[n].inputs[tnet->modules[n].ninputs++] = get_node_inputnames(node, tnet, 0);
                    tnet->modules[n].inputs[tnet->modules[n].ninputs++] = get_node_inputnames(node, tnet, 1);
                    thload_BatchNorm(tnet->modules + n, NULL, NULL, NULL, NULL, 0, 1);
                }
                else if(node->kind() == aten::batch_norm){
                //aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled)
                    float* weight = NULL;
                    int s = 0;
                    at::Tensor bb;
                    if(value_to_ivalue.find(node->input(1)) != value_to_ivalue.end()){
                        bb = get_tensor(&value_to_ivalue[node->input(1)]);
                        s = bb.sizes()[0];
                        weight = (float*) bb.data_ptr();
                    }
                    float* bias = NULL;
                    if(value_to_ivalue.find(node->input(2)) != value_to_ivalue.end()){
                        bb = get_tensor(&value_to_ivalue[node->input(2)]);
                        if(s != bb.sizes()[0])
                            s = bb.sizes()[0];
                        bias = (float*) bb.data_ptr();
                    }
                    float* run_mean = NULL;
                    if(value_to_ivalue.find(node->input(3)) != value_to_ivalue.end()){
                        bb = get_tensor(&value_to_ivalue[node->input(3)]);
                        if(s != bb.sizes()[0])
                            s = bb.sizes()[0];
                        run_mean = (float*) bb.data_ptr();
                    }
                    float* run_var = NULL;
                    if(value_to_ivalue.find(node->input(4)) != value_to_ivalue.end()){
                        bb = get_tensor(&value_to_ivalue[node->input(4)]);
                        if(s != bb.sizes()[0])
                            s = bb.sizes()[0];
                        run_var = (float*) bb.data_ptr();
                    }
                    int eps = get_const_double(node->input(7));
                    thload_BatchNorm(tnet->modules + n, weight, bias, run_mean, run_var, eps, s);
                }
                else if(node->kind() == aten::relu)
                    thload_Threshold(tnet->modules + n);
                else if(node->kind() == aten::tanh)
                    thload_Tanh(tnet->modules + n);
                else if(node->kind() == aten::sigmoid)
                    thload_Sigmoid(tnet->modules + n);
                else if(node->kind() == aten::view)
                    thload_View(tnet->modules + n);
                else if(node->kind() == aten::max_pool2d || node->kind() == aten::avg_pool2d){
                //aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False)
                //aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None)
                    int kW, kH, dW, dH, pW, pH, dlW, dlH, ceil;
                    kW = get_const_intlist(node->input(1))[0];
                    kH = get_const_intlist(node->input(1))[1];
                    if(!get_const_intlist(node->input(2)).empty()){
                        dW = get_const_intlist(node->input(2))[0];
                        dH = get_const_intlist(node->input(2))[1];
                    }
                    else{
                        dW = kW;
                        dH = kH;
                    }
                    pW = get_const_intlist(node->input(3))[0];
                    pH = get_const_intlist(node->input(3))[1];
                    if(node->kind() == aten::max_pool2d){
                        dlW = get_const_intlist(node->input(4))[0];
                        dlH = get_const_intlist(node->input(4))[1];
                        ceil = get_const_bool(node->input(5));
                        thload_Maxpool2d(tnet->modules + n, kW, kH, pW, pH, dW, dH, dlW, dlH, ceil);
                    }
                    else{
                        ceil = get_const_bool(node->input(4));
                        thload_Avgpool2d(tnet->modules + n, kW, kH, pW, pH, dW, dH, ceil);
                    }
                }

                //connect layers with input output ids
                if(num_input == 1)//one input layers
                    tnet->modules[n].inputs[tnet->modules[n].ninputs++] = get_node_inputnames(node, tnet, 0);
                tnet->modules[n].output = thnets::THNTensor_new(thnets::DT_FLOAT);
                tnet->modules[n].outputname[0] = strdup(std::to_string(node->output(0)->unique()).c_str());
                tnet->modules[n].net = tnet;
                tnet->nelem = ++n;

                //BatchNorm absorb
                if(tnet->nelem > 0 && tnet->modules[tnet->nelem-1].type == thnets::MT_SpatialBatchNormalization && tnet->modules[tnet->nelem-1].inputs[0] >= 0) {
                    thnets::module *prevm = &tnet->modules[tnet->modules[tnet->nelem-1].inputs[0]];
                    if((prevm->type == thnets::MT_Dropout || prevm->type == thnets::MT_Transpose) && prevm->inputs[0] >= 0)
                        prevm = &tnet->modules[prevm->inputs[0]];
                    if(prevm->type == thnets::MT_SpatialConvolutionVirtMM ||
                        prevm->type == thnets::MT_SpatialFullConvolution ||
                        prevm->type == thnets::MT_Linear)
                    {
                        if(count_output_used(prevm->outputname[0]) == 1) {
                            absorb_bn(tnet, tnet->nelem-1, prevm);
                            n--;
                        }
                    }
                }

            }
        }
        assert(in_node);
        char image[100];
        int inW = 1, inH = 1, inP = 1, inZ = 1;
        {//get input size
            assert (value_to_ivalue.find(in_node->input(0)) != value_to_ivalue.end());
            in_data = get_tensor(&value_to_ivalue[in_node->input(0)]);
            inP = in_data.sizes()[1];
            if(in_data.sizes().size() == 4) {//WxHxPxB
                inH = in_data.sizes()[2];
                inW = in_data.sizes()[3];
                sprintf(image,"%dx%dx%d", inW, inH, inP);
            }
            else if(in_data.sizes().size() == 5) {//WxHxZxPxB
                inZ = in_data.sizes()[2];
                inH = in_data.sizes()[3];
                inW = in_data.sizes()[4];
                sprintf(image,"%dx%dx%dx%d", inW, inH, inZ, inP);
            }
            else if (in_data.sizes().size() == 1){//P
                inP = in_data.sizes()[0];
                sprintf(image,"%dx%dx%d", 1, 1, inP);
            }
            else{//WxPxB
                inW = in_data.sizes()[2];
                sprintf(image,"%dx%dx%d", inW, 1, inP);
            }
        }
        net->net = tnet;
        if(!debug_.empty())
            ie_setflag(cmem, "debug", debug_.c_str());
        if(!options_.empty())
            ie_setflag(cmem, "options", options_.c_str());
        uint64_t swoutsize[MAX_INPUTS];
        unsigned *noutdims;
        unsigned noutputs = 0;
        uint64_t **outshapes;
        //pass THNETWORK to thnets2lst to create lst
        ext_thnets2lst(cmem, net, image, -1, 1);
        //ie_compile: skip onnx parser if already lst exist and modelpath="$keeplst"
        if(clusters_ != 1){
            char s[300];
            sprintf(s, "%d", clusters_);
            ie_setflag(cmem, "nclusters", s);
        }
        cmem = ie_compile(cmem, "$keeplst", "save.bin", image, &noutputs, &noutdims, &outshapes);

        std::vector< std::vector<int64_t> > osizes;
        std::vector<int64_t> tsizes;// batch, plane, Z, H, W
        for(unsigned i = 0; i < noutputs; i++){
            for(unsigned j = 0; j < noutdims[i]; j++){
                tsizes.push_back(outshapes[i][j]);
            }
            osizes.push_back(tsizes);
            tsizes.clear();
        }
        cmem = ie_init(cmem, "save.bin", &noutputs, &noutdims, &outshapes, first ? 0 : fcmem);
        fcmem = cmem;
        ie_getinfo(cmem, "addr_off", &laddr_off, sizeof(laddr_off));

        cinfo = new Compiled_info(cmem);
        cinfo->osizes = osizes;
        cinfo->input_elements[0] = inP*inH*inW;
        for(int i = 0; i < noutputs; i++) {
            cinfo->output_elements[i] = swoutsize[i];
        }
        cinfo->in_node = in_node;
        cache_[spec] = cinfo;
    }
    else{
        cinfo = cache_[spec];
    }
    assert(cinfo);
    //---------------------------------------------------------------------------------------------
    float* out = (float*) malloc(cinfo->output_elements[0]*sizeof(float));

    //get input tensor from Stack using node->input Value to select the correct IValue
    assert (value_to_ivalue.find(cinfo->in_node->input(0)) != value_to_ivalue.end());
    in_data = get_tensor(&value_to_ivalue[cinfo->in_node->input(0)]);
    float* data = (float*) in_data.data_ptr();
    int err = 0;

    err = ie_run(cinfo->cmem, &data, cinfo->input_elements, ninputs, &out, cinfo->output_elements, cinfo->osizes.size());
    if (err)
        fprintf(stderr,"ie_run ERROR %d\n", err);

    drop(stack, num_inputs);//remove input IValues from Stack
    int oidx = 0;
    for (auto& output : subgraph_->outputs()) {
        //add output tensors to Stack for following subgraphs or nodes in the model
        auto tensor = at::from_blob(out, cinfo->osizes[oidx++]);
        auto var = torch::autograd::make_variable(tensor);
        stack->push_back(IValue(var));
    }
}


