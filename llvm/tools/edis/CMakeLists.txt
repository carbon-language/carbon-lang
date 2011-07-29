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

# TODO: Process EnhancedDisassembly.exports

if( NOT WIN32 AND LLVM_ENABLE_PIC )
  set(bsl ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS ON)
  add_llvm_library(EnhancedDisassembly ${SOURCES})
  add_llvm_library_dependencies(EnhancedDisassembly ${EDIS_DEPENDS})
  set_property(TARGET EnhancedDisassembly PROPERTY
    OUTPUT_NAME "EnhancedDisassembly")
  set(BUILD_SHARED_LIBS ${bsl})
  set(EnhancedDisassembly_STATIC_TARGET_NAME EnhancedDisassembly_static)
else()
  set(EnhancedDisassembly_STATIC_TARGET_NAME EnhancedDisassembly)
endif()

if( NOT BUILD_SHARED_LIBS )
  add_llvm_library(${EnhancedDisassembly_STATIC_TARGET_NAME} ${SOURCES})
  set_property(TARGET ${EnhancedDisassembly_STATIC_TARGET_NAME} PROPERTY
    OUTPUT_NAME "EnhancedDisassembly")

  add_llvm_library_dependencies(${EnhancedDisassembly_STATIC_TARGET_NAME}
    ${EDIS_DEPENDS})
    
endif()
