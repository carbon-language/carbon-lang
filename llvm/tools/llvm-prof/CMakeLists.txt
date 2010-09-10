add_llvm_tool(llvm-prof
  llvm-prof.cpp
  )

target_link_libraries(llvm-prof LLVMBitReader LLVMAnalysis)
