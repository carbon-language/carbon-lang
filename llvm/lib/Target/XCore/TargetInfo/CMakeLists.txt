add_llvm_library(LLVMXCoreInfo
  XCoreTargetInfo.cpp
  )

add_dependencies(LLVMXCoreInfo XCoreCommonTableGen)
