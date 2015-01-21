add_llvm_library(lldELF
  ArrayOrderPass.cpp
  ELFLinkingContext.cpp
  Reader.cpp
  Writer.cpp
  LINK_LIBS
    lldHexagonELFTarget
    lldMipsELFTarget
    lldPPCELFTarget
    lldPasses
    lldReaderWriter
    lldX86ELFTarget
    lldX86_64ELFTarget
    lldAArch64ELFTarget
    lldARMELFTarget
  )

include_directories(.)

add_subdirectory(X86)
add_subdirectory(X86_64)
add_subdirectory(PPC)
add_subdirectory(Mips)
add_subdirectory(Hexagon)
add_subdirectory(AArch64)
add_subdirectory(ARM)
