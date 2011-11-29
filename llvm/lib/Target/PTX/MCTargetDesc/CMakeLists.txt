add_llvm_library(LLVMPTXDesc
  PTXMCTargetDesc.cpp
  PTXMCAsmInfo.cpp
  )

add_dependencies(LLVMPTXDesc PTXCommonTableGen)
