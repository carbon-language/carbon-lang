include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMPTXInfo
  PTXTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMPTXInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMPTXInfo PTXCommonTableGen)
