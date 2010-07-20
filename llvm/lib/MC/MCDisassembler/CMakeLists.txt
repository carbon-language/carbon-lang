
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc
  COMMAND ${LLVM_TABLEGEN_EXE} -o ${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc
          -gen-enhanced-disassembly-header
  DEPENDS tblgen
  COMMENT "Building enhanced disassembly semantic information header (EDInfo.inc)")
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc PROPERTIES GENERATED 1)

add_llvm_library(LLVMMCDisassembler
  EDDisassembler.cpp
  EDOperand.cpp
  EDInst.cpp
  EDToken.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/EDInfo.inc
  )
