include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMMipsInfo
  MipsTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMMipsInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMMipsInfo MipsCommonTableGen)
