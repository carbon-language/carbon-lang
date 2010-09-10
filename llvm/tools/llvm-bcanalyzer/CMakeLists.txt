set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-bcanalyzer
  llvm-bcanalyzer.cpp
  )

target_link_libraries(llvm-bcanalyzer LLVMBitReader)
