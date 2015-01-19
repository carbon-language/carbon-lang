set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  ScalarOpts
  Support
  native
  mcjit
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch5
  toy.cpp
  )
