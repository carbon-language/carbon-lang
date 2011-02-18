add_llvm_library(LLVMExecutionEngine
  ExecutionEngine.cpp
  ExecutionEngineBindings.cpp
  )

add_subdirectory(Interpreter)
add_subdirectory(JIT)
add_subdirectory(MCJIT)
