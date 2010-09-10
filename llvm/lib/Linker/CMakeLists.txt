add_llvm_library(LLVMLinker
  LinkArchives.cpp
  LinkItems.cpp
  LinkModules.cpp
  Linker.cpp
  )

target_link_libraries(LLVMLinker
  LLVMArchive
  LLVMBitReader
  LLVMCore
  LLVMTransformUtils
  )
