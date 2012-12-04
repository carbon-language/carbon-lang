add_lld_library(lldMachO
  WriterMachO.cpp
  WriterOptionsMachO.cpp
  ReferenceKinds.cpp
  )

target_link_libraries(lldMachO
  lldReaderWriter
  lldPasses
  )
