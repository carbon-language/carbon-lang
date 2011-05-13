# TODO: Support other architectures. See Makefile.
add_definitions(-DENABLE_X86_JIT)

add_llvm_library(LLVMJIT
  Intercept.cpp
  JIT.cpp
  JITDebugRegisterer.cpp
  JITDwarfEmitter.cpp
  JITEmitter.cpp
  JITMemoryManager.cpp
  OProfileJITEventListener.cpp
  )
