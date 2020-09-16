PYTORCH_DIR=$(python3 -c 'import os, torch; print(os.path.dirname(os.path.realpath(torch.__file__)))')
mkdir -p build && cd build
cmake .. -DPYTORCH_DIR=${PYTORCH_DIR}
make -j
