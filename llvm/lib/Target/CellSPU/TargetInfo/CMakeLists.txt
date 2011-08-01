include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMCellSPUInfo
  CellSPUTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMCellSPUInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMCellSPUInfo CellSPUCommonTableGen)
