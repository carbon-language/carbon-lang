add_llvm_library(LLVMLinker
  LinkArchives.cpp
  LinkItems.cpp
  LinkModules.cpp
  Linker.cpp
  )

add_llvm_library_dependencies(LLVMLinker
  LLVMArchive
  LLVMBitReader
  LLVMCore
  LLVMSupport
  LLVMTransformUtils
  )
