set(LLVM_NO_RTTI 1)

include_directories(.)
include_directories(../Linux)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessPOSIX
  POSIXStopInfo.cpp
  POSIXThread.cpp
  ProcessMessage.cpp
  ProcessPOSIX.cpp
  ProcessPOSIXLog.cpp
  RegisterContextPOSIX_mips64.cpp
  RegisterContextPOSIX_x86.cpp
  RegisterContextPOSIXProcessMonitor_mips64.cpp
  RegisterContextPOSIXProcessMonitor_x86.cpp
  RegisterContextFreeBSD_i386.cpp
  RegisterContextFreeBSD_mips64.cpp
  RegisterContextFreeBSD_x86_64.cpp
  RegisterContextLinux_i386.cpp
  RegisterContextLinux_x86_64.cpp
  )
