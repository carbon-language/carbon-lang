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

add_llvm_library_dependencies(LLVMJIT
  LLVMCore
  LLVMExecutionEngine
  LLVMRuntimeDyld
  LLVMSupport
  LLVMTarget
  )
