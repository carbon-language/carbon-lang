add_llvm_tool(llvm-mc
  llvm-mc.cpp
  Disassembler.cpp
  )

add_llvm_link_components(llvm-mc all)
target_link_libraries(llvm-mc LLVMMC LLVMMCParser LLVMMCDisassembler)
