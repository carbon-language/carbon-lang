include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMARMAsmPrinter
  ARMAsmPrinter.cpp
  ARMInstPrinter.cpp
  ARMMCInstLower.cpp
  )
add_dependencies(LLVMARMAsmPrinter ARMCodeGenTable_gen)
