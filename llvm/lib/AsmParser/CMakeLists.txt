# AsmParser, with a Bison generated parser:
include(FindBison)
find_bison()

IF(BISON_EXECUTABLE)
bison_generator(
  "llvmAsm"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.y"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.cpp"
  )
ELSE(BISON_EXECUTABLE)
  MESSAGE(STATUS "Bison not found, the pregenerated files will be used")

  IF(WIN32)
    FILE(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR} WIN_SOURCE_DIR)

    ADD_CUSTOM_COMMAND(
      OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.cpp"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.cpp.cvs"
      COMMAND copy \"${WIN_SOURCE_DIR}\\llvmAsmParser.cpp.cvs\" \"${WIN_SOURCE_DIR}\\llvmAsmParser.cpp\"
    )
    ADD_CUSTOM_COMMAND(
      OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.h"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/llvmAsmParser.h.cvs"
      COMMAND copy \"${WIN_SOURCE_DIR}\\llvmAsmParser.h.cvs\" \"${WIN_SOURCE_DIR}\\llvmAsmParser.h\"
    )
  ELSE(WIN32)
    MESSAGE(SEND_ERROR "Not implemented")
    # fixme do the same with cp instead of copy
  ENDIF(WIN32)
ENDIF(BISON_EXECUTABLE)

add_llvm_library(LLVMAsmParser
  llvmAsmParser.cpp  # Generated from llvmAsmParser.y or copyied from corresponding .cvs
  llvmAsmParser.h    # needed for the associated custom command to work
  LLLexer.cpp
  Parser.cpp
  )
