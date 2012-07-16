include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMAMDGPUInfo
  AMDGPUTargetInfo.cpp
  )

add_dependencies(LLVMAMDGPUInfo AMDGPUCommonTableGen intrinsics_gen)
