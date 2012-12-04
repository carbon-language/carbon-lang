add_lld_library(lldPECOFF
  ReaderCOFF.cpp
  WriterPECOFF.cpp
  )

target_link_libraries(lldPECOFF
  lldReaderWriter
  )
