include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMX86Disassembler
  X86Disassembler.cpp
  )
add_dependencies(LLVMX86Disassembler X86CodeGenTable_gen)
