add_llvm_library(LLVMNVPTXInfo
  NVPTXTargetInfo.cpp
  )

add_dependencies(LLVMNVPTXInfo NVPTXCommonTableGen)
