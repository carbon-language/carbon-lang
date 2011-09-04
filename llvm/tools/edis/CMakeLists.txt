include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(SOURCES
  ../../include/llvm-c/EnhancedDisassembly.h
  EDMain.cpp
  )

set(EDIS_DEPENDS LLVMMCDisassembler LLVMMCParser)
if( LLVM_TARGETS_TO_BUILD MATCHES X86 )
  list(APPEND EDIS_DEPENDS LLVMX86AsmPrinter LLVMX86AsmParser LLVMX86Disassembler LLVMX86Desc)
endif()
if( LLVM_TARGETS_TO_BUILD MATCHES ARM )
  list(APPEND EDIS_DEPENDS LLVMARMAsmPrinter LLVMARMAsmParser LLVMARMDisassembler LLVMARMDesc)
endif()

add_llvm_library(EnhancedDisassembly ${SOURCES})
set_property(TARGET EnhancedDisassembly PROPERTY
  OUTPUT_NAME "EnhancedDisassembly")

add_llvm_library_dependencies(EnhancedDisassembly
  ${EDIS_DEPENDS})
