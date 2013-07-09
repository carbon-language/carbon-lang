add_llvm_library(LLVMInstrumentation
  AddressSanitizer.cpp
  BoundsChecking.cpp
  DebugIR.cpp
  EdgeProfiling.cpp
  GCOVProfiling.cpp
  MemorySanitizer.cpp
  Instrumentation.cpp
  OptimalEdgeProfiling.cpp
  PathProfiling.cpp
  ProfilingUtils.cpp
  ThreadSanitizer.cpp
  )

add_dependencies(LLVMInstrumentation intrinsics_gen)
