add_llvm_library(LLVMipa
  CallGraph.cpp
  CallGraphSCCPass.cpp
  FindUsedTypes.cpp
  GlobalsModRef.cpp
  IPA.cpp
  )

add_llvm_library_dependencies(LLVMipa
  LLVMAnalysis
  LLVMCore
  LLVMSupport
  )
