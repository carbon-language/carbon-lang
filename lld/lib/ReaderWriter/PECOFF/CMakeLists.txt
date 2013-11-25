add_lld_library(lldPECOFF
  IdataPass.cpp
  PECOFFLinkingContext.cpp
  ReaderCOFF.cpp
  ReaderImportHeader.cpp
  WriterPECOFF.cpp
  )

target_link_libraries(lldPECOFF
  lldReaderWriter
  lldDriver
  )
