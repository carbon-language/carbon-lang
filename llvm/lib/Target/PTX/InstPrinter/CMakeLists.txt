include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMPTXAsmPrinter
  PTXInstPrinter.cpp
  )

add_dependencies(LLVMPTXAsmPrinter PTXCommonTableGen)

add_llvm_library_dependencies(LLVMPTXAsmPrinter
  LLVMMC
  LLVMSupport
  )

