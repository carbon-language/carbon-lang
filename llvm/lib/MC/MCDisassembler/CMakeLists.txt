add_llvm_library(LLVMMCDisassembler
  EDDisassembler.cpp
  EDOperand.cpp
  EDInst.cpp
  EDToken.cpp
  )

add_llvm_link_components(LLVMMCDisassembler all)
