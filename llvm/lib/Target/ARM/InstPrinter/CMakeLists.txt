add_llvm_library(LLVMARMAsmPrinter
  ARMInstPrinter.cpp
  )

add_dependencies(LLVMARMAsmPrinter ARMCommonTableGen)
