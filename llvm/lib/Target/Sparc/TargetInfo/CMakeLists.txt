add_llvm_library(LLVMSparcInfo
  SparcTargetInfo.cpp
  )

add_dependencies(LLVMSparcInfo SparcCommonTableGen)
