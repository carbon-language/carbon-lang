add_llvm_library(LLVMR600Info
  AMDGPUTargetInfo.cpp
  )

add_dependencies(LLVMR600Info AMDGPUCommonTableGen intrinsics_gen)
