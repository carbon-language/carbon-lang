add_llvm_library(LLVMMCDisassembler
  Disassembler.cpp
  EDDisassembler.cpp
  EDInst.cpp
  EDOperand.cpp
  EDToken.cpp
  )

add_llvm_library_dependencies(LLVMMCDisassembler
  LLVMMC
  LLVMMCParser
  LLVMSupport
  )
