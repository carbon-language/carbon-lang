set(LLVM_LINK_COMPONENTS object)

add_subdirectory(ELF)
add_subdirectory(MachO)
add_subdirectory(Native)
add_subdirectory(PECOFF)
add_subdirectory(YAML)
add_lld_library(lldReaderWriter
  Reader.cpp
  Writer.cpp
  ReaderArchive.cpp
  )

target_link_libraries(lldReaderWriter
  lldCore
  )
