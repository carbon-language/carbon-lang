add_llvm_library(LLVMInstrumentation
  AddressSanitizer.cpp
  BoundsChecking.cpp
  DataFlowSanitizer.cpp
  GCOVProfiling.cpp
  MemorySanitizer.cpp
  Instrumentation.cpp
  InstrProfiling.cpp
  SanitizerCoverage.cpp
  ThreadSanitizer.cpp
  )

add_dependencies(LLVMInstrumentation intrinsics_gen)
