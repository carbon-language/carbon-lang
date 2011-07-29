add_llvm_library(LLVMXCoreDesc
  XCoreMCTargetDesc.cpp
  XCoreMCAsmInfo.cpp
  )

add_llvm_library_dependencies(LLVMXCoreDesc
  LLVMMC
  LLVMXCoreInfo
  )

add_dependencies(LLVMXCoreDesc XCoreCommonTableGen)

# Hack: we need to include 'main' target directory to grab private headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/..)
