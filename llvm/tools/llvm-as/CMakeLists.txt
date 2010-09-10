set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-as
  llvm-as.cpp
  )

target_link_libraries(llvm-as LLVMAsmParser LLVMBitWriter)
