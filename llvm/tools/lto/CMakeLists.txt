set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LTO
  MC
  MCDisassembler
  Support
  )

set(SOURCES
  LTODisassembler.cpp
  lto.cpp
  )

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/lto.exports)

add_llvm_library(LTO SHARED ${SOURCES})

install(FILES ${LLVM_MAIN_INCLUDE_DIR}/llvm-c/lto.h
  DESTINATION include/llvm-c)
