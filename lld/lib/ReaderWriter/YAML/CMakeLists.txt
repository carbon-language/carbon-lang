add_lld_library(lldYAML
  ReaderWriterYAML.cpp
  )

target_link_libraries(lldYAML
  lldReaderWriter
  )
