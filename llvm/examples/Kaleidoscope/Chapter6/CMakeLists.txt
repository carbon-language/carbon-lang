set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  JIT
  ScalarOpts
  Support
  nativecodegen
  )

add_llvm_example(Kaleidoscope-Ch6
  toy.cpp
  )
