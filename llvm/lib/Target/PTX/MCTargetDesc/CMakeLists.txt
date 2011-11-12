add_llvm_library(LLVMPTXDesc
  PTXMCTargetDesc.cpp
  PTXMCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMPTXDesc
  LLVMMC
  LLVMPTXAsmPrinter
  LLVMPTXInfo
  LLVMSupport
  )

add_dependencies(LLVMPTXDesc PTXCommonTableGen)
