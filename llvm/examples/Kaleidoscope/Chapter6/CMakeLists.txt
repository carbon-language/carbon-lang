set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  MC
  ScalarOpts
  Support
  nativecodegen
  mcjit
  )

add_llvm_example(Kaleidoscope-Ch6
  toy.cpp
  )
