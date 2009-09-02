include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMX86AsmPrinter
  X86ATTAsmPrinter.cpp
  X86ATTInstPrinter.cpp
  X86AsmPrinter.cpp
  X86MCInstLower.cpp
  X86IntelAsmPrinter.cpp
  )
add_dependencies(LLVMX86AsmPrinter X86CodeGenTable_gen)
