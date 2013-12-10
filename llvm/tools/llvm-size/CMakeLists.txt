set(LLVM_LINK_COMPONENTS
  Object
  Support
  )

add_llvm_tool(llvm-size
  llvm-size.cpp
  )
