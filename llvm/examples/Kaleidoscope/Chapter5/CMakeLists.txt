set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  MCJIT
  ScalarOpts
  Support
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch5
  toy.cpp
  )
