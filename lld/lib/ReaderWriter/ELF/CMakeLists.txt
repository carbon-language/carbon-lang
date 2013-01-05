add_lld_library(lldELF
  HexagonReference.cpp
  PPCReference.cpp
  ReaderELF.cpp
  ReferenceKinds.cpp
  WriterELF.cpp
  WriterOptionsELF.cpp
  X86Reference.cpp
  X86_64Reference.cpp
  )

target_link_libraries(lldELF
  lldReaderWriter
  )
