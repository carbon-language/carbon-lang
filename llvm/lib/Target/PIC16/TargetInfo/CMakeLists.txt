include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMPIC16Info
  PIC16TargetInfo.cpp
  )

add_dependencies(LLVMPIC16Info PIC16Table_gen)
