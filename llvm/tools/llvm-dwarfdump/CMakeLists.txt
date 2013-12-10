set(LLVM_LINK_COMPONENTS
  DebugInfo
  Object
  Support
  )

add_llvm_tool(llvm-dwarfdump
  llvm-dwarfdump.cpp
  )
