# AsmParser, with a Bison generated parser:
include(FindBison)
find_bison()
bison_generator(
  "llvmAsm"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.y"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.cpp"
  )

add_llvm_library(LLVMAsmParser
  llvmAsmParser.cpp  # Generated from llvmAsmParser.y
  LLLexer.cpp
  Parser.cpp
  )
