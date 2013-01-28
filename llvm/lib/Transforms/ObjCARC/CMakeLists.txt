add_llvm_library(LLVMObjCARCOpts
  ObjCARC.cpp
  ObjCARCOpts.cpp
  ObjCARCExpand.cpp
  ObjCARCAPElim.cpp
  ObjCARCAliasAnalysis.cpp
  )

add_dependencies(LLVMObjCARCOpts intrinsics_gen)
