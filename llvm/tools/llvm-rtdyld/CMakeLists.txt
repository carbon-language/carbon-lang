set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfo
  ExecutionEngine
  MC
  RuntimeDyld
  Support
  )

add_llvm_tool(llvm-rtdyld
  llvm-rtdyld.cpp
  )
