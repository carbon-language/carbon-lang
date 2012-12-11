include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMR600Info
  AMDGPUTargetInfo.cpp
  )

add_dependencies(LLVMR600Info AMDGPUCommonTableGen intrinsics_gen)
