add_llvm_library(LLVMAArch64AsmParser
  AArch64AsmParser.cpp
  )

add_dependencies(LLVMAArch64AsmParser AArch64CommonTableGen)
