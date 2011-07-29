include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. 
                     ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMMBlazeAsmPrinter
  MBlazeInstPrinter.cpp
  )

add_llvm_library_dependencies(LLVMMBlazeAsmPrinter
  LLVMMC
  LLVMSupport
  )

add_dependencies(LLVMMBlazeAsmPrinter MBlazeCommonTableGen)
