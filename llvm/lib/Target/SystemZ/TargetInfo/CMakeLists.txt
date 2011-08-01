include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMSystemZInfo
  SystemZTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMSystemZInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMSystemZInfo SystemZCommonTableGen)
