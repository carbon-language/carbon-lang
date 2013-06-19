add_lld_library(lldPECOFF
  Atoms.cpp
  PECOFFTargetInfo.cpp
  ReaderCOFF.cpp
  ReaderImportHeader.cpp
  WriterPECOFF.cpp
  )

target_link_libraries(lldPECOFF
  lldReaderWriter
  )
