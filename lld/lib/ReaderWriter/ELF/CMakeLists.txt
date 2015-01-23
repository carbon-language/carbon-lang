add_llvm_library(lldELF
  ArrayOrderPass.cpp
  ELFLinkingContext.cpp
  Reader.cpp
  Writer.cpp
  LINK_LIBS
    lldCore
    lldPasses
    lldYAML
    LLVMSupport
  )

include_directories(.)

add_subdirectory(X86)
add_subdirectory(X86_64)
add_subdirectory(PPC)
add_subdirectory(Mips)
add_subdirectory(Hexagon)
add_subdirectory(AArch64)
add_subdirectory(ARM)
