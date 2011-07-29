add_llvm_library(LLVMExecutionEngine
  ExecutionEngine.cpp
  ExecutionEngineBindings.cpp
  TargetSelect.cpp
  )

add_llvm_library_dependencies(LLVMExecutionEngine
  LLVMCore
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_subdirectory(Interpreter)
add_subdirectory(JIT)
add_subdirectory(MCJIT)
add_subdirectory(RuntimeDyld)
