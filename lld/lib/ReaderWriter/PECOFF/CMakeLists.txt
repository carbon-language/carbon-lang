add_lld_library(lldPECOFF
  EdataPass.cpp
  IdataPass.cpp
  PECOFFLinkingContext.cpp
  Pass.cpp
  ReaderCOFF.cpp
  ReaderImportHeader.cpp
  WriterPECOFF.cpp
  )

target_link_libraries(lldPECOFF
  lldReaderWriter
  lldDriver
  )
