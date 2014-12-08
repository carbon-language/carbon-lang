set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  MC
  ScalarOpts
  Support
  native
  mcjit
  )

add_llvm_example(Kaleidoscope-Ch5
  toy.cpp
  )
