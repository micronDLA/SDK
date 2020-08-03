/*
The process is:
1- a pytorch model is created in .py
2- torchscript will trace/script this model. Trace/script is run the model and creates a operations graph
3- add a pass to label parts of this graph as custom operation to be run differently
4- define how to run this new labeled subgraph
*/
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include "compiler.h"

namespace py = pybind11;
using namespace torch;
using namespace torch::jit;

static bool mdla_enabled = false;
static std::string debug = "";
static std::string opts = "";
static int clusters = 1;

void registerMDLAOp(){
    //TODO: add layers and add passes to combine graphs
    // Register a pass to convert the IR into one with our operator
    // it labels ops with new symbol in graph to be executed using the new symbol
    torch::jit::RegisterPass pass([](std::shared_ptr<Graph>& g) {
            if (mdla_enabled) {
                FuseLinear(g);//input to linear tensor must be dim=1
                FuseSupportedOps(g);
            }
    });

    // We are only dealing with pure operations (no aliasing or in place mutation), so our subgraph will always be pure
    auto options = c10::OperatorOptions();
    options.setAliasAnalysis(AliasAnalysisKind::PURE_FUNCTION);

    // Register a custom compiler/implementation for our subgraph
    torch::jit::RegisterOperators op({
            torch::jit::Operator(
                    getMDLASymbol(),
                    [](const torch::jit::Node* node) -> torch::jit::Operation {
                    auto compiler = std::make_shared<MDLACompiler>(node, debug, opts, clusters);
                    return [compiler](Stack& stack) {
                    compiler->run(stack);
                    return 0;
                    };
                    },
                    options)});

}

//creates python module torchMDLA that will register our custom pass and operation
PYBIND11_MODULE(torchMDLA, m) {
    registerMDLAOp();
    m.def("enable",
    [](std::string debug_, std::string options_, int clusters_) {
        debug = debug_;
        opts = options_;
        clusters = clusters_;
        mdla_enabled = true;
    },
    py::arg("debug") = "",
    py::arg("options") = "",
    py::arg("clusters") = 1 );
    m.def("disable", []() { mdla_enabled = false; });
}
