include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMSparcInfo
  SparcTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMSparcInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMSparcInfo SparcCommonTableGen)
