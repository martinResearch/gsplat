#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/
# NOTE: CUDA_HOME must be a Windows-format path (C:\...) because it's
# consumed by native Python (via PyTorch's CUDAExtension).  Git Bash does
# NOT auto-translate env var values, only command-line arguments.

case ${1} in
  cu126)
    export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6"
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin:$PATH"
    export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
    ;;
  cu124)
    export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin:$PATH"
    export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
    ;;
  cu121)
    export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin:$PATH"
    export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
    ;;
  cu118)
    export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8"
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin:$PATH"
    export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
    ;;
  cu117)
    export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7"
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin:$PATH"
    export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
    ;;
  cu116)
    export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6"
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin:$PATH"
    export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
    ;;
  cu115)
    export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5"
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin:$PATH"
    export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
    ;;
  cu113)
    export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3"
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/bin:$PATH"
    export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
    ;;
  *)
    ;;
esac