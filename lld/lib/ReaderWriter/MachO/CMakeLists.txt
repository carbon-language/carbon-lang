add_lld_library(lldMachO
  GOTPass.cpp
  MachOLinkingContext.cpp
  MachONormalizedFileBinaryReader.cpp
  MachONormalizedFileBinaryWriter.cpp
  MachONormalizedFileFromAtoms.cpp
  MachONormalizedFileToAtoms.cpp
  MachONormalizedFileYAML.cpp
  ReferenceKinds.cpp
  StubsPass.cpp
  WriterMachO.cpp
  )

target_link_libraries(lldMachO
  lldReaderWriter
  lldPasses
  )

include_directories(.)
