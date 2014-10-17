add_lld_library(lldMachO
  ArchHandler.cpp
  ArchHandler_arm.cpp
  ArchHandler_arm64.cpp
  ArchHandler_x86.cpp
  ArchHandler_x86_64.cpp
  CompactUnwindPass.cpp
  GOTPass.cpp
  MachOLinkingContext.cpp
  MachONormalizedFileBinaryReader.cpp
  MachONormalizedFileBinaryWriter.cpp
  MachONormalizedFileFromAtoms.cpp
  MachONormalizedFileToAtoms.cpp
  MachONormalizedFileYAML.cpp
  ShimPass.cpp
  StubsPass.cpp
  WriterMachO.cpp
  )

target_link_libraries(lldMachO ${cmake_2_8_12_INTERFACE}
  lldReaderWriter
  lldPasses
  )

include_directories(.)
