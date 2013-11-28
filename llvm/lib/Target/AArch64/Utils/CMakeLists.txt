add_llvm_library(LLVMAArch64Utils
  AArch64BaseInfo.cpp
  )

add_dependencies(LLVMAArch64Utils AArch64CommonTableGen)
