include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMARMAsmParser
  ARMAsmLexer.cpp
  ARMAsmParser.cpp
  )

add_dependencies(LLVMARMAsmParser ARMCommonTableGen)

add_llvm_library_dependencies(LLVMARMAsmParser
  LLVMARMDesc
  LLVMARMInfo
  LLVMMC
  LLVMMCParser
  LLVMSupport
  )
