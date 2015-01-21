add_llvm_library(lldYAML
  ReaderWriterYAML.cpp
  LINK_LIBS
    lldReaderWriter
    LLVMSupport
  )
