add_llvm_library(LLVMArchive
  Archive.cpp
  ArchiveReader.cpp
  ArchiveWriter.cpp
  )

add_llvm_library_dependencies(LLVMArchive
  LLVMBitReader
  LLVMCore
  LLVMSupport
  )
