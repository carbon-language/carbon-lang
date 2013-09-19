add_llvm_tool(llvm-lto
  llvm-lto.cpp
  )

target_link_libraries(llvm-lto LTO LLVMSupport)

add_dependencies(llvm-lto lto)
