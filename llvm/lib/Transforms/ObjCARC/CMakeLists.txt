add_llvm_library(LLVMObjCARCOpts
  ObjCARC.cpp
  ObjCARCOpts.cpp
  )

add_dependencies(LLVMObjCARCOpts intrinsics_gen)
