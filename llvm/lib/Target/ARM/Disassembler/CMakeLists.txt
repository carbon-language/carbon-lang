add_llvm_library(LLVMARMDisassembler
  ARMDisassembler.cpp
  )
# workaround for hanging compilation on MSVC8, 9 and 10
if( MSVC_VERSION EQUAL 1400 OR MSVC_VERSION EQUAL 1500 OR MSVC_VERSION EQUAL 1600 )
set_property(
  SOURCE ARMDisassembler.cpp
  PROPERTY COMPILE_FLAGS "/Od"
  )
endif()
