add_lld_library(lldNative
  ReaderNative.cpp
  WriterNative.cpp
  )

target_link_libraries(lldNative
  lldReaderWriter
  )
