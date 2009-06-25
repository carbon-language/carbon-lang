include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMPowerPCAsmPrinter
  PPCAsmPrinter.cpp
  )
add_dependencies(LLVMPowerPCAsmPrinter PowerPCCodeGenTable_gen)