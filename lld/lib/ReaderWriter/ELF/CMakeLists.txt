add_lld_library(lldELF
  ArrayOrderPass.cpp
  ELFLinkingContext.cpp
  Reader.cpp
  Writer.cpp
  )

target_link_libraries(lldELF ${cmake_2_8_12_INTERFACE}
  lldHexagonELFTarget
  lldMipsELFTarget
  lldPPCELFTarget
  lldPasses
  lldReaderWriter
  lldX86ELFTarget
  lldX86_64ELFTarget
  lldAArch64ELFTarget
  )

include_directories(.)

add_subdirectory(AArch64)
add_subdirectory(Hexagon)
add_subdirectory(Mips)
add_subdirectory(PPC)
add_subdirectory(X86)
add_subdirectory(X86_64)
