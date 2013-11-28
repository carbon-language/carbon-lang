add_llvm_library(LLVMNVPTXAsmPrinter
  NVPTXInstPrinter.cpp
  )

add_dependencies(LLVMNVPTXAsmPrinter NVPTXCommonTableGen)
