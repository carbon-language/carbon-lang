include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMX86AsmParser
  X86AsmLexer.cpp
  X86AsmParser.cpp
  )

add_llvm_library_dependencies(LLVMX86AsmParser
  LLVMMC
  LLVMMCParser
  LLVMSupport
  LLVMX86Desc
  LLVMX86Info
  )

add_dependencies(LLVMX86AsmParser X86CommonTableGen)
