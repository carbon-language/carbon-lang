add_llvm_library(LLVMX86Info
  X86TargetInfo.cpp
  )

add_dependencies(LLVMX86Info X86CommonTableGen)
