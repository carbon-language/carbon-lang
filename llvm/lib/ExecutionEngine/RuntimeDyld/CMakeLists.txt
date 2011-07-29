add_llvm_library(LLVMRuntimeDyld
  RuntimeDyld.cpp
  RuntimeDyldMachO.cpp
  )

add_llvm_library_dependencies(LLVMRuntimeDyld
  LLVMObject
  LLVMSupport
  )
