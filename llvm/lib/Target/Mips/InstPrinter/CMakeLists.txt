add_llvm_library(LLVMMipsAsmPrinter
  MipsInstPrinter.cpp
  )

add_dependencies(LLVMMipsAsmPrinter MipsCommonTableGen)
