set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  MCJIT
  Object
  RuntimeDyld
  ScalarOpts
  Support
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch6
  toy.cpp
  )
