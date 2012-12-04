add_lld_library(lldYAML
  YamlKeyValues.cpp
  ReaderYAML.cpp
  WriterYAML.cpp
  )

target_link_libraries(lldYAML
  lldReaderWriter
  )
