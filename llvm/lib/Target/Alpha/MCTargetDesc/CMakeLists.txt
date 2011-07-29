add_llvm_library(LLVMAlphaDesc
  AlphaMCTargetDesc.cpp
  AlphaMCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMAlphaDesc
  LLVMAlphaInfo
  LLVMMC
  )

add_dependencies(LLVMAlphaDesc AlphaCommonTableGen)
