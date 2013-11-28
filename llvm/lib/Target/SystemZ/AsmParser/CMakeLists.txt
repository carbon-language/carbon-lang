add_llvm_library(LLVMSystemZAsmParser
  SystemZAsmParser.cpp
  )

add_dependencies(LLVMSystemZAsmParser SystemZCommonTableGen)
