add_llvm_library(LLVMXCoreDesc
  XCoreMCTargetDesc.cpp
  XCoreMCAsmInfo.cpp
  )

add_dependencies(LLVMXCoreDesc XCoreCommonTableGen)
