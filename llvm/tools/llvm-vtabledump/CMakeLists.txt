set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Object
  Support
  )

add_llvm_tool(llvm-vtabledump
  llvm-vtabledump.cpp
  Error.cpp
  )
