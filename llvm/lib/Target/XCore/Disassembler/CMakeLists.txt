add_llvm_library(LLVMXCoreDisassembler
  XCoreDisassembler.cpp
  )

add_dependencies(LLVMXCoreDisassembler XCoreCommonTableGen)
