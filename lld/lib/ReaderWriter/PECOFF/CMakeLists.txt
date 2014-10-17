add_lld_library(lldPECOFF
  EdataPass.cpp
  IdataPass.cpp
  LinkerGeneratedSymbolFile.cpp
  LoadConfigPass.cpp
  PECOFFLinkingContext.cpp
  Pass.cpp
  ReaderCOFF.cpp
  ReaderImportHeader.cpp
  WriterImportLibrary.cpp
  WriterPECOFF.cpp
  )

target_link_libraries(lldPECOFF ${cmake_2_8_12_INTERFACE}
  lldReaderWriter
  lldDriver
  )
