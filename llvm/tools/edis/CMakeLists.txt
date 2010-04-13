set(LLVM_NO_RTTI 1)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc
  COMMAND ${LLVM_TABLEGEN_EXE} -o ${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc
          -gen-enhanced-disassembly-header ${CMAKE_CURRENT_SOURCE_DIR}/EDInfo.td
  DEPENDS tblgen
  COMMENT "Building enhanced disassembly semantic information header (EDInfo.inc)")
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc PROPERTIES GENERATED 1)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_llvm_library(EnhancedDisassembly
  EDDisassembler.cpp
  EDInst.cpp
  EDMain.cpp
  EDOperand.cpp
  EDToken.cpp
  ../../include/llvm-c/EnhancedDisassembly.h
  ${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc
)

set_target_properties(EnhancedDisassembly
  PROPERTIES
  LINKER_LANGUAGE CXX)

