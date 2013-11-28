add_llvm_library(LLVMAArch64Disassembler
  AArch64Disassembler.cpp
  )

add_dependencies(LLVMAArch64Disassembler AArch64CommonTableGen)
