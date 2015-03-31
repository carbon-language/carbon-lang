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

set(LLVM_REQUIRES_RTTI 1)

set(LLVM_BUILD_EXAMPLES OFF)

add_kaleidoscope_chapter(Kaleidoscope-Ch7
  toy.cpp
  )
