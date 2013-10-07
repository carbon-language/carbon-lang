set(LLVM_LINK_COMPONENTS object)

add_subdirectory(ELF)
add_subdirectory(MachO)
add_subdirectory(Native)
add_subdirectory(PECOFF)
add_subdirectory(YAML)
add_lld_library(lldReaderWriter
  CoreLinkingContext.cpp
  LinkerScript.cpp
  Reader.cpp
  ReaderLinkerScript.cpp
  Writer.cpp
  )

target_link_libraries(lldReaderWriter
  lldCore
  lldPasses
  )
