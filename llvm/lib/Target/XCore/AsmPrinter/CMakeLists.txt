include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMXCoreAsmPrinter
  XCoreAsmPrinter.cpp
  )
add_dependencies(LLVMXCoreAsmPrinter XCoreCodeGenTable_gen)
