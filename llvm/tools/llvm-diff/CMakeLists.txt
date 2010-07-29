set(LLVM_LINK_COMPONENTS support asmparser bitreader)

add_llvm_tool(llvm-diff
  llvm-diff.cpp
  DifferenceEngine.cpp
  )
