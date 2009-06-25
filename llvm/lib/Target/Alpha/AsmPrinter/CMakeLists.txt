include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMAlphaAsmPrinter
  AlphaAsmPrinter.cpp
  )
add_dependencies(LLVMAlphaAsmPrinter AlphaCodeGenTable_gen)
