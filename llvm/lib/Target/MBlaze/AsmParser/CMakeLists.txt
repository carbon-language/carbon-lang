include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. 
                     ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMMBlazeAsmParser
  MBlazeAsmLexer.cpp
  MBlazeAsmParser.cpp
  )

add_llvm_library_dependencies(LLVMMBlazeAsmParser
  LLVMMBlazeInfo
  LLVMMC
  LLVMMCParser
  LLVMSupport
  )

add_dependencies(LLVMMBlazeAsmParser MBlazeCommonTableGen)
