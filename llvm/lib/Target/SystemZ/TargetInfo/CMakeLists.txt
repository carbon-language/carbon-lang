add_llvm_library(LLVMSystemZInfo
  SystemZTargetInfo.cpp
  )

add_dependencies(LLVMSystemZInfo SystemZCommonTableGen)
