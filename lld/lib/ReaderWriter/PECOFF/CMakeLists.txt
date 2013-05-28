add_lld_library(lldPECOFF
  PECOFFTargetInfo.cpp
  ReaderCOFF.cpp
  WriterPECOFF.cpp
  )

target_link_libraries(lldPECOFF
  lldReaderWriter
  )
