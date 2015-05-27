add_llvm_library(LLVMMIRParser
  MIRParser.cpp
  )

add_dependencies(LLVMMIRParser intrinsics_gen)
