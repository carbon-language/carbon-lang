add_llvm_tool(llvm-nm
  llvm-nm.cpp
  )

target_link_libraries(llvm-nm LLVMArchive LLVMBitReader)
