add_llvm_library(LLVMXCoreAsmPrinter
  XCoreInstPrinter.cpp
  )

add_dependencies(LLVMXCoreAsmPrinter XCoreCommonTableGen)
