add_llvm_tool(llvm-link
  llvm-link.cpp
  )

target_link_libraries(llvm-link
  LLVMLinker
  LLVMBitReader
  LLVMBitWriter
  LLVMAsmParser
  )
