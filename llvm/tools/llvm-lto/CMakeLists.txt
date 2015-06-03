set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  LTO
  MC
  Support
  Target
  )

add_llvm_tool(llvm-lto
  llvm-lto.cpp
  )

