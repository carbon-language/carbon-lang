add_llvm_library(lldNative
  ReaderNative.cpp
  WriterNative.cpp
  LINK_LIBS
    lldCore
    LLVMSupport
  )
