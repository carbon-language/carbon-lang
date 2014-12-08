set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  MC
  ScalarOpts
  Support
  TransformUtils
  nativecodegen
  mcjit
  )

set(LLVM_REQUIRES_RTTI 1)

add_llvm_example(Kaleidoscope-Ch7
  toy.cpp
  )
