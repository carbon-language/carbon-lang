set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LTO
  MC
  MCDisassembler
  Support
  )

add_definitions( -DLLVM_VERSION_INFO=\"${PACKAGE_VERSION}\" )

set(SOURCES
  LTODisassembler.cpp
  lto.cpp
  )

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/lto.exports)

if(NOT CYGWIN AND LLVM_ENABLE_PIC)
  set(ENABLE_SHARED SHARED)
endif()

add_llvm_library(LTO ${ENABLE_SHARED} STATIC ${SOURCES})

install(FILES ${LLVM_MAIN_INCLUDE_DIR}/llvm-c/lto.h
  DESTINATION include/llvm-c)
