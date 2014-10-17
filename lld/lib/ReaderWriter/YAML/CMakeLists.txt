add_lld_library(lldYAML
  ReaderWriterYAML.cpp
  )

target_link_libraries(lldYAML ${cmake_2_8_12_INTERFACE}
  lldReaderWriter
  )
