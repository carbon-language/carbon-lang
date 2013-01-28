add_llvm_library(LLVMObjCARCOpts
  ObjCARC.cpp
  ObjCARCOpts.cpp
  ObjCARCExpand.cpp
  ObjCARCAPElim.cpp
  ObjCARCAliasAnalysis.cpp
  ObjCARCUtil.cpp
  )

add_dependencies(LLVMObjCARCOpts intrinsics_gen)
