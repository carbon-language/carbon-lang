set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Object
  OrcJIT
  RuntimeDyld
  Support
  native
  )

add_kaleidoscope_chapter(Kaleidoscope-Orc-fully_lazy
  toy.cpp
  )
