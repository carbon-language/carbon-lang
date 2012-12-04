add_lld_library(lldELF
  ReaderELF.cpp
  WriterELF.cpp
  ReferenceKinds.cpp
  HexagonReference.cpp
  PPCReference.cpp
  X86Reference.cpp
  WriterOptionsELF.cpp
  )

target_link_libraries(lldELF
  lldReaderWriter
  )
