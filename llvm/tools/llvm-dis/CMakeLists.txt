set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-dis
  llvm-dis.cpp
  )

target_link_libraries(llvm-dis LLVMCore LLVMBitReader)
