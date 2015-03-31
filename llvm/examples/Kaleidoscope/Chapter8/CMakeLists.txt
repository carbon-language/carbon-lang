set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MCJIT
  RuntimeDyld
  Support
  native
  )

set(LLVM_REQUIRES_RTTI 1)

set(LLVM_BUILD_EXAMPLES OFF)

add_kaleidoscope_chapter(Kaleidoscope-Ch8
  toy.cpp
  )
