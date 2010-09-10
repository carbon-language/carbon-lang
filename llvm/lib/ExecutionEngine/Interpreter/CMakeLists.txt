add_llvm_library(LLVMInterpreter
  Execution.cpp
  ExternalFunctions.cpp
  Interpreter.cpp
  )

target_link_libraries(LLVMInterpreter
  LLVMCodeGen
  LLVMCore
  LLVMExecutionEngine
  LLVMTarget
  )
