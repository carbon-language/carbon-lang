add_llvm_library(LLVMPTXDesc
  PTXMCTargetDesc.cpp
  PTXMCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMPTXDesc
  LLVMMC
  LLVMPTXInfo
  LLVMPTXAsmPrinter
  LLVMSupport
  )

add_dependencies(LLVMPTXDesc PTXCommonTableGen)
