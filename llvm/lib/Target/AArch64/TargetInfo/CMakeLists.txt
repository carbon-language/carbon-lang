add_llvm_library(LLVMAArch64Info
  AArch64TargetInfo.cpp
  )

add_dependencies(LLVMAArch64Info AArch64CommonTableGen)
