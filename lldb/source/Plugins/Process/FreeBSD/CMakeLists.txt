set(LLVM_NO_RTTI 1)

include_directories(.)
include_directories(../POSIX)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessFreeBSD
  ProcessFreeBSD.cpp
  FreeBSDSignals.cpp
  FreeBSDThread.cpp
  ProcessMonitor.cpp
  )
