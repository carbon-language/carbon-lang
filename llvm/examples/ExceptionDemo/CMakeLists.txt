set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MC
  MCJIT
  Support
  nativecodegen
  )

# Enable EH and RTTI for this demo
set(LLVM_REQUIRES_EH 1)
set(LLVM_REQUIRES_RTTI 1)

add_llvm_example(ExceptionDemo
  ExceptionDemo.cpp
  )

export_executable_symbols(ExceptionDemo)
