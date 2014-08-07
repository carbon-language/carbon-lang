set(LLVM_LINK_COMPONENTS
  BitWriter
  Core
  ExecutionEngine
  JIT
  MC
  Support
  nativecodegen
  )

add_llvm_example(BrainF
  BrainF.cpp
  BrainFDriver.cpp
  )
