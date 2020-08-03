#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/argument_spec.h>
#include <ATen/ATen.h>
#include "fusion_pass.h"
#include "api.h"

using namespace torch::jit;

#define MAX_INPUTS 20

class Compiled_info{
    public:
        void* cmem;
        uint64_t input_elements[MAX_INPUTS];
        uint64_t output_elements[MAX_INPUTS];
        std::vector< std::vector<int64_t> > osizes;
        Node* in_node;
        Compiled_info(void* cm):cmem(cm){};
        ~Compiled_info(){
            ie_free(cmem);
        };
};

class MDLACompiler {
    public:
        MDLACompiler(const torch::jit::Node* node,
        std::string debug = "",
        std::string options = "",
        int clusters = 1)
        : subgraph_(node->g(torch::jit::attr::Subgraph)),
        debug_(debug) {
            g_size = 0;
            clusters_ = clusters;
            for (auto node : subgraph_->nodes())
                g_size++;
            options_ = options;
            if(clusters > 1 && options.find("C") == std::string::npos)
                options_.append("C");
        };
        ~MDLACompiler(){
            for(auto it : cache_)
                delete it.second;
        };
    //subgraph is the custom function that the fusion_pass created "mdla::CompilationGroup"
    //implementation of the custom subgraph
    //Stack is the tensor inputs of the subgraph. The output is added to the Stack
        void run(torch::jit::Stack& stack);
        int count_output_used(const char *name);
        int g_size;//graph size
    private:
        std::string debug_;//debug flags
        std::string options_;//options flags
        int clusters_;//number of clusters to use
        std::shared_ptr<torch::jit::Graph> subgraph_;//subgraph with node to be run
        std::unordered_map<torch::jit::CompleteArgumentSpec, Compiled_info*> cache_;//cache what has been compiled
};
