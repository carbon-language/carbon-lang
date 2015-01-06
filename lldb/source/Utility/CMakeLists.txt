set(LLVM_NO_RTTI 1)

add_lldb_library(lldbUtility
  ARM_DWARF_Registers.cpp
  ARM64_DWARF_Registers.cpp
  KQueue.cpp
  PseudoTerminal.cpp
  Range.cpp
  SharingPtr.cpp
  StringExtractor.cpp
  StringExtractorGDBRemote.cpp
  StringLexer.cpp
  TimeSpecTimeout.cpp
  UriParser.cpp
  )
