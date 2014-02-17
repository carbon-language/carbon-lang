set(LLVM_LINK_COMPONENTS core support )

add_llvm_tool(llvm-profdata
  llvm-profdata.cpp
  )
