add_llvm_library(LLVMMIRParser
  MIParser.cpp
  MIRParser.cpp
  )

add_dependencies(LLVMMIRParser intrinsics_gen)
