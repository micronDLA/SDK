#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

void FuseSupportedOps(std::shared_ptr<torch::jit::Graph> graph);

const torch::jit::Symbol& getMDLASymbol();
bool canHandle(const torch::jit::Node* node);
