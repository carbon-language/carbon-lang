include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMPowerPCAsmPrinter
  PPCInstPrinter.cpp
  )

add_llvm_library_dependencies(LLVMPowerPCAsmPrinter
  LLVMMC
  LLVMSupport
  )

add_dependencies(LLVMPowerPCAsmPrinter PowerPCCommonTableGen)
