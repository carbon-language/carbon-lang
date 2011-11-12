include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMCppBackendInfo
  CppBackendTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMCppBackendInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )
