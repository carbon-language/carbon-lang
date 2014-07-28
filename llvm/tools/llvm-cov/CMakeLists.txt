set(LLVM_LINK_COMPONENTS core support )

add_llvm_tool(llvm-cov
  llvm-cov.cpp
  gcov.cpp
  )
