set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Object
  RuntimeDyld
  Support
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Orc-lazy_irgen
  toy.cpp
  )
