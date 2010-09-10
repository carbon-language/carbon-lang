add_llvm_tool(llvm-ld
  Optimize.cpp
  llvm-ld.cpp
  )

target_link_libraries(llvm-ld
  LLVMipo
  LLVMScalarOpts
  LLVMLinker
  LLVMArchive
  LLVMBitWriter
  )
