add_llvm_library(LLVMSystemZDisassembler
  SystemZDisassembler.cpp
  )

add_dependencies(LLVMSystemZDisassembler SystemZCommonTableGen)
