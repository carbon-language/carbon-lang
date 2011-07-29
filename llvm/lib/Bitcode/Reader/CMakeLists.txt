add_llvm_library(LLVMBitReader
  BitReader.cpp
  BitcodeReader.cpp
  )

add_llvm_library_dependencies(LLVMBitReader
  LLVMCore
  LLVMSupport
  )
