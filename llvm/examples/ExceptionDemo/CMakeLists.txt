set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MC
  MCJIT
  RuntimeDyld
  Support
  nativecodegen
  )

# Enable EH and RTTI for this demo
set(LLVM_REQUIRES_EH 1)
set(LLVM_REQUIRES_RTTI 1)

set(LLVM_BUILD_EXAMPLES OFF)

add_llvm_example(ExceptionDemo
  ExceptionDemo.cpp
  )

export_executable_symbols(ExceptionDemo)
