add_lld_library(lldMachO
  MachOLinkingContext.cpp
  MachONormalizedFileBinaryReader.cpp
  MachONormalizedFileBinaryWriter.cpp
  MachONormalizedFileFromAtoms.cpp
  MachONormalizedFileYAML.cpp
  ReferenceKinds.cpp
  WriterMachO.cpp
  )

target_link_libraries(lldMachO
  lldReaderWriter
  lldPasses
  )
