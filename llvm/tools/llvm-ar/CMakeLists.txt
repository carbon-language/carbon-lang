set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-ar
  llvm-ar.cpp
  )

target_link_libraries(llvm-ar LLVMArchive)

# TODO: Support check-local.
