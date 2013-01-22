add_lld_library(lldMachO
  MachOTargetInfo.cpp
  WriterMachO.cpp
  WriterOptionsMachO.cpp
  ReferenceKinds.cpp
  )

target_link_libraries(lldMachO
  lldReaderWriter
  lldPasses
  )
