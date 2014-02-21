set(LLVM_LINK_COMPONENTS
  Object
  Support
  )

add_llvm_tool(llvm-nm
  llvm-nm.cpp
  )
