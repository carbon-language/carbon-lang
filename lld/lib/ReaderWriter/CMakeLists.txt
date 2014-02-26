add_subdirectory(ELF)
add_subdirectory(MachO)
add_subdirectory(Native)
add_subdirectory(PECOFF)
add_subdirectory(YAML)

set(LLVM_LINK_COMPONENTS object)

add_lld_library(lldReaderWriter
  CoreLinkingContext.cpp
  FileArchive.cpp
  LinkerScript.cpp
  Reader.cpp
  Writer.cpp
  FileArchive.cpp
  )

target_link_libraries(lldReaderWriter ${cmake_2_8_12_INTERFACE}
  lldCore
  lldPasses
  )
