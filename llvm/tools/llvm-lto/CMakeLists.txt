set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LTO
  MC
  Support
  )

add_llvm_tool(llvm-lto
  llvm-lto.cpp
  )

