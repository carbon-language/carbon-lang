add_llvm_library(LLVMBitWriter
  BitWriter.cpp
  BitcodeWriter.cpp
  BitcodeWriterPass.cpp
  ValueEnumerator.cpp
  )

add_llvm_library_dependencies(LLVMBitWriter
  LLVMCore
  LLVMSupport
  )
