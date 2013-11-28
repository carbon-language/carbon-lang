add_llvm_library(LLVMMipsInfo
  MipsTargetInfo.cpp
  )

add_dependencies(LLVMMipsInfo MipsCommonTableGen)
