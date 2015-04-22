set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  ExecutionEngine
  InstCombine
  MCJIT
  RuntimeDyld
  ScalarOpts
  Support
  TransformUtils
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch7
  toy.cpp
  )
