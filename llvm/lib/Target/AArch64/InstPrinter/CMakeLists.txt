add_llvm_library(LLVMAArch64AsmPrinter
  AArch64InstPrinter.cpp
  )

add_dependencies(LLVMAArch64AsmPrinter AArch64CommonTableGen)

