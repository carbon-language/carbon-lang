include_directories(.)
include_directories(../POSIX)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessFreeBSD
  ProcessFreeBSD.cpp
  FreeBSDThread.cpp
  ProcessMonitor.cpp

  POSIXStopInfo.cpp
  RegisterContextPOSIXProcessMonitor_arm.cpp
  RegisterContextPOSIXProcessMonitor_arm64.cpp
  RegisterContextPOSIXProcessMonitor_powerpc.cpp
  RegisterContextPOSIXProcessMonitor_x86.cpp
  RegisterContextPOSIXProcessMonitor_mips64.cpp
  )
