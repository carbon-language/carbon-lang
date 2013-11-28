add_llvm_library(LLVMR600AsmPrinter
  AMDGPUInstPrinter.cpp
  )

add_dependencies(LLVMR600AsmPrinter AMDGPUCommonTableGen)
