set(LLVM_LINK_COMPONENTS support object bitreader)

add_llvm_tool(llvm-ar
  llvm-ar.cpp
  )

# TODO: Support check-local.
