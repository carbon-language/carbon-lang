add_llvm_library(LLVMARMInfo
  ARMTargetInfo.cpp
  )

add_dependencies(LLVMARMInfo ARMCommonTableGen)
