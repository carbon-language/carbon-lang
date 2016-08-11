set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LTO
  Object
  Support
  Target
  )

add_llvm_tool(llvm-lto2
  llvm-lto2.cpp
  )
