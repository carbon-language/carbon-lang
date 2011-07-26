add_llvm_library(LLVMAlphaDesc
  AlphaMCTargetDesc.cpp
  AlphaMCAsmInfo.cpp
  )
add_dependencies(LLVMAlphaDesc AlphaCommonTableGen)
