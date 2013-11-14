add_lld_library(lldPECOFF
  PECOFFLinkingContext.cpp
  ReaderCOFF.cpp
  ReaderImportHeader.cpp
  WriterPECOFF.cpp
  )

target_link_libraries(lldPECOFF
  lldReaderWriter
  lldDriver
  )
