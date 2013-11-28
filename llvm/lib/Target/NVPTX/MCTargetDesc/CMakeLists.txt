add_llvm_library(LLVMNVPTXDesc
  NVPTXMCAsmInfo.cpp
  NVPTXMCTargetDesc.cpp
  )

add_dependencies(LLVMNVPTXDesc NVPTXCommonTableGen)
