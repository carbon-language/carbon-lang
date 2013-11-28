add_llvm_library(LLVMARMAsmParser
  ARMAsmParser.cpp
  )

add_dependencies(LLVMARMAsmParser ARMCommonTableGen)
