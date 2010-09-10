add_llvm_library(LLVMInstrumentation
  EdgeProfiling.cpp
  OptimalEdgeProfiling.cpp
  ProfilingUtils.cpp
  )

target_link_libraries(LLVMInstrumentation
  LLVMAnalysis
  LLVMCore
  LLVMTransformUtils
  )
