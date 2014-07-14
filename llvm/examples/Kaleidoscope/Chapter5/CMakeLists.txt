set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  JIT
  MC
  ScalarOpts
  Support
  nativecodegen
  )

add_llvm_example(Kaleidoscope-Ch5
  toy.cpp
  )
