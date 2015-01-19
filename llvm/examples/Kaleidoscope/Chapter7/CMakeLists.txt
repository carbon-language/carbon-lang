set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  MCJIT
  ScalarOpts
  Support
  TransformUtils
  native
  )

set(LLVM_REQUIRES_RTTI 1)

add_kaleidoscope_chapter(Kaleidoscope-Ch7
  toy.cpp
  )
