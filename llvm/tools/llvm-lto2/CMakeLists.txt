set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  LTO
  Object
  Support
  )

add_llvm_tool(llvm-lto2
  llvm-lto2.cpp
  )
