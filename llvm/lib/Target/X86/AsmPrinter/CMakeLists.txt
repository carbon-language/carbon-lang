include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMX86AsmPrinter
  X86ATTAsmPrinter.cpp
  X86AsmPrinter.cpp
  X86IntelAsmPrinter.cpp
  )
