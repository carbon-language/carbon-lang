add_llvm_tool(lli
  lli.cpp
  )

add_llvm_link_components(lli jit interpreter nativecodegen)
target_link_libraries(lli LLVMBitReader LLVMSelectionDAG)
