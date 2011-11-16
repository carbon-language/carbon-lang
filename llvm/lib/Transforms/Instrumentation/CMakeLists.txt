add_llvm_library(LLVMInstrumentation
  AddressSanitizer.cpp
  EdgeProfiling.cpp
  GCOVProfiling.cpp
  Instrumentation.cpp
  OptimalEdgeProfiling.cpp
  PathProfiling.cpp
  ProfilingUtils.cpp
  )

add_llvm_library_dependencies(LLVMInstrumentation
  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMTransformUtils
  )
