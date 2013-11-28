add_llvm_library(LLVMPowerPCAsmParser
  PPCAsmParser.cpp
  )

add_dependencies(LLVMPowerPCAsmParser PowerPCCommonTableGen)
