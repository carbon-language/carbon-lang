add_llvm_tool(llc
  llc.cpp
  )

add_llvm_link_components(llc all)
target_link_libraries(llc LLVMBitReader LLVMAsmParser)
