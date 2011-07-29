include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMAlphaInfo
  AlphaTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMAlphaInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMAlphaInfo AlphaCommonTableGen)
