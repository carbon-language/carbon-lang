add_llvm_library(LLVMExecutionEngine
  ExecutionEngine.cpp
  ExecutionEngineBindings.cpp
  )

target_link_libraries(LLVMExecutionEngine LLVMCore LLVMTarget)
