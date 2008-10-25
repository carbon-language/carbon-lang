# AsmParser, with a Bison generated parser:
include(FindBison)
find_bison()

if(BISON_EXECUTABLE)
  bison_generator(
    "llvmAsm"
    "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.y"
    "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.cpp"
    )
else(BISON_EXECUTABLE)
  message(STATUS "Bison not found, the pregenerated files will be used")

  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.cpp.cvs"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.cpp.cvs" "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.cpp"
    )
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.h.cvs"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.h.cvs" "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.h"
    )
endif(BISON_EXECUTABLE)

add_llvm_library(LLVMAsmParser
  llvmAsmParser.cpp  # Generated from llvmAsmParser.y or copyied from corresponding .cvs
  llvmAsmParser.h    # needed for the associated custom command to work
  LLLexer.cpp
  Parser.cpp
  )
