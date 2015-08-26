set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MCJIT
  Object
  RuntimeDyld
  Support
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Ch8
  toy.cpp
  )
