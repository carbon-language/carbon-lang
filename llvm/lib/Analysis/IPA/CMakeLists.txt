add_llvm_library(LLVMipa
  CallGraph.cpp
  CallGraphSCCPass.cpp
  FindUsedTypes.cpp
  GlobalsModRef.cpp
  )

target_link_libraries(LLVMipa LLVMAnalysis LLVMCore)
