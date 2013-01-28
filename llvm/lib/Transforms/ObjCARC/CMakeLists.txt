add_llvm_library(LLVMObjCARCOpts
  ObjCARC.cpp
  ObjCARCOpts.cpp
  ObjCARCExpand.cpp
  )

add_dependencies(LLVMObjCARCOpts intrinsics_gen)
