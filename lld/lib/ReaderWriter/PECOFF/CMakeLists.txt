add_lld_library(lldPECOFF
  EdataPass.cpp
  IdataPass.cpp
  LoadConfigPass.cpp
  PECOFFLinkingContext.cpp
  Pass.cpp
  ReaderCOFF.cpp
  ReaderImportHeader.cpp
  WriterImportLibrary.cpp
  WriterPECOFF.cpp
  )

target_link_libraries(lldPECOFF
  lldReaderWriter
  lldDriver
  )
