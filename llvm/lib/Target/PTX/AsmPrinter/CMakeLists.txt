include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMPTXAsmPrinter
  PTXAsmPrinter.cpp
  )
add_dependencies(LLVMPTXAsmPrinter PTXCodeGenTable_gen)
