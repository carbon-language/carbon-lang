add_llvm_library(LLVMArchive
  Archive.cpp
  ArchiveReader.cpp
  ArchiveWriter.cpp
  )

target_link_libraries(LLVMArchive LLVMBitReader LLVMCore)
