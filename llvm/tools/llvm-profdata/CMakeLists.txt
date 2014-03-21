set(LLVM_LINK_COMPONENTS core profiledata support )

add_llvm_tool(llvm-profdata
  llvm-profdata.cpp
  )
