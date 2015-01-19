set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  ScalarOpts
  Support
  TransformUtils
  native
  mcjit
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch4
  toy.cpp
  )
