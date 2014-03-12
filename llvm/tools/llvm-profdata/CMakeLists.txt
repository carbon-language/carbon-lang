set(LLVM_LINK_COMPONENTS core profile support)

add_llvm_tool(llvm-profdata
  llvm-profdata.cpp
  )
