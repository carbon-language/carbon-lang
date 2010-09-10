add_llvm_library(LLVMBitReader
  BitReader.cpp
  BitcodeReader.cpp
  )

target_link_libraries(LLVMBitReader LLVMCore)
