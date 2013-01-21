add_llvm_library(LLVMipa
  CallGraph.cpp
  CallGraphSCCPass.cpp
  CallPrinter.cpp
  FindUsedTypes.cpp
  GlobalsModRef.cpp
  IPA.cpp
  InlineCost.cpp
  )

add_dependencies(LLVMipa intrinsics_gen)
