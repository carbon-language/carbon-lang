set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MCJIT
  Support
  native
  )

set(LLVM_REQUIRES_RTTI 1)

add_kaleidoscope_chapter(Kaleidoscope-Ch8
  toy.cpp
  )
