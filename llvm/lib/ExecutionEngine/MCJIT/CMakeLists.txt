add_llvm_library(LLVMMCJIT
  MCJIT.cpp
  Intercept.cpp
  )

add_llvm_library_dependencies(LLVMMCJIT
  LLVMCore
  LLVMExecutionEngine
  LLVMRuntimeDyld
  LLVMSupport
  LLVMTarget
  )
