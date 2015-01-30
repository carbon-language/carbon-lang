set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  Object
  Support
  )

add_llvm_tool(llvm-dwarfdump
  llvm-dwarfdump.cpp
  )
