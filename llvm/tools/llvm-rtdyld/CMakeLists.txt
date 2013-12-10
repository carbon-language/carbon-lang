set(LLVM_LINK_COMPONENTS
  DebugInfo
  ExecutionEngine
  RuntimeDyld
  Support
  )

add_llvm_tool(llvm-rtdyld
  llvm-rtdyld.cpp
  )
