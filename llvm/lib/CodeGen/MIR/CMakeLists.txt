add_llvm_library(LLVMMIR
  MIRPrinter.cpp
  MIRPrintingPass.cpp
  MIRParser.cpp
  )

add_dependencies(LLVMMIR intrinsics_gen)
