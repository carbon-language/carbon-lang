add_llvm_library(lldELF
  DynamicFile.cpp
  ELFFile.cpp
  ELFLinkingContext.cpp
  FileCommon.cpp
  Reader.cpp
  Writer.cpp
  LINK_LIBS
    lldReaderWriter
    lldCore
    lldYAML
    LLVMSupport
  )

include_directories(.)

add_subdirectory(X86)
add_subdirectory(X86_64)
add_subdirectory(Mips)
add_subdirectory(Hexagon)
add_subdirectory(AArch64)
add_subdirectory(ARM)
