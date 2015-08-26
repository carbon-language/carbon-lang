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
  TransformUtils
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch4
  toy.cpp
  )
