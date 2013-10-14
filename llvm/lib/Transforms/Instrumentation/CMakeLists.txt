add_llvm_library(LLVMInstrumentation
  AddressSanitizer.cpp
  BoundsChecking.cpp
  DataFlowSanitizer.cpp
  DebugIR.cpp
  GCOVProfiling.cpp
  MemorySanitizer.cpp
  Instrumentation.cpp
  ThreadSanitizer.cpp
  )

add_dependencies(LLVMInstrumentation intrinsics_gen)
