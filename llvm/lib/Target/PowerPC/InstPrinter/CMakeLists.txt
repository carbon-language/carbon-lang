add_llvm_library(LLVMPowerPCAsmPrinter
  PPCInstPrinter.cpp
  )

add_dependencies(LLVMPowerPCAsmPrinter PowerPCCommonTableGen)
