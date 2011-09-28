set(LLVM_LINK_COMPONENTS instrumentation )

add_llvm_tool(llvm-cov
  GCOVReader.cpp
  llvm-cov.cpp
  )
