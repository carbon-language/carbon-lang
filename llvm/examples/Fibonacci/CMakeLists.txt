set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Interpreter
  JIT
  Support
  nativecodegen
  )

add_llvm_example(Fibonacci
  fibonacci.cpp
  )
