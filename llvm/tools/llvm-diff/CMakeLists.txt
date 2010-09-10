add_llvm_tool(llvm-diff
  llvm-diff.cpp
  DifferenceEngine.cpp
  )

target_link_libraries(llvm-diff LLVMAsmParser LLVMBitReader)
