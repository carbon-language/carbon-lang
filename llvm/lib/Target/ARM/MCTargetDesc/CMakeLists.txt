add_llvm_library(LLVMARMDesc
  ARMAsmBackend.cpp
  ARMMCAsmInfo.cpp
  ARMMCCodeEmitter.cpp
  ARMMCExpr.cpp
  ARMMCTargetDesc.cpp
  ARMMachObjectWriter.cpp
  )
add_dependencies(LLVMARMDesc ARMCommonTableGen)

# Hack: we need to include 'main' target directory to grab private headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/..)

add_llvm_library_dependencies(LLVMARMDesc
  LLVMARMAsmPrinter
  LLVMARMInfo
  LLVMMC
  LLVMSupport
  )
