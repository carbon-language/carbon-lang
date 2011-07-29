add_llvm_library(LLVMSystemZDesc
  SystemZMCTargetDesc.cpp
  SystemZMCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMSystemZDesc
  LLVMMC
  LLVMSystemZInfo
  )

add_dependencies(LLVMSystemZDesc SystemZCommonTableGen)

# Hack: we need to include 'main' target directory to grab private headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/..)
