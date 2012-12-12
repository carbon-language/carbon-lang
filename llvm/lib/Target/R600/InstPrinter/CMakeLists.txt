include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMR600AsmPrinter
  AMDGPUInstPrinter.cpp
  )

add_dependencies(LLVMR600AsmPrinter AMDGPUCommonTableGen)
