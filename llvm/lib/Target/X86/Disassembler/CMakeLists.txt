include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMX86Disassembler
  X86Disassembler.cpp
  X86DisassemblerDecoder.c
  )
# workaround for hanging compilation on MSVC9
if( MSVC_VERSION EQUAL 1500 )
set_property(
  SOURCE X86Disassembler.cpp
  PROPERTY COMPILE_FLAGS "/Od"
  )
endif()
add_dependencies(LLVMX86Disassembler X86CodeGenTable_gen)
