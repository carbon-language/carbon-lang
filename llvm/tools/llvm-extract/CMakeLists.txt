add_llvm_tool(llvm-extract
  llvm-extract.cpp
  )

target_link_libraries(llvm-extract
  LLVMAsmParser
  LLVMipo
  LLVMBitReader
  LLVMBitWriter
  )
