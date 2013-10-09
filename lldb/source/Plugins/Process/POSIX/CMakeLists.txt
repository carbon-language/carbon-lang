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
  RegisterContextPOSIX_i386.cpp
  RegisterContextPOSIX_x86_64.cpp
  RegisterContextFreeBSD_i386.cpp
  RegisterContextFreeBSD_x86_64.cpp
  RegisterContextLinux_i386.cpp
  RegisterContextLinux_x86_64.cpp
  RegisterContextPOSIXProcessMonitor_i386.cpp
  RegisterContextPOSIXProcessMonitor_x86_64.cpp
  )
