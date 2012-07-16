
add_llvm_library(LLVMAMDGPUDesc
  AMDGPUMCTargetDesc.cpp
  AMDGPUMCAsmInfo.cpp
  )

add_dependencies(LLVMAMDGPUDesc AMDGPUCommonTableGen)
