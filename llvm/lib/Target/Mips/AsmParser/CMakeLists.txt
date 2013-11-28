add_llvm_library(LLVMMipsAsmParser
  MipsAsmParser.cpp
  )

add_dependencies(LLVMMipsAsmParser MipsCommonTableGen)
