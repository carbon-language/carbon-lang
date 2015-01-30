set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoDWARF
  ExecutionEngine
  MC
  Object
  RuntimeDyld
  Support
  )

add_llvm_tool(llvm-rtdyld
  llvm-rtdyld.cpp
  )
