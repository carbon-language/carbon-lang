set(LLVM_NO_RTTI 1)

include_directories(.)
include_directories(../Linux)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessPOSIX
  CrashReason.cpp
  POSIXStopInfo.cpp
  POSIXThread.cpp
  ProcessMessage.cpp
  ProcessPOSIX.cpp
  ProcessPOSIXLog.cpp
  RegisterContextPOSIXProcessMonitor_arm64.cpp
  RegisterContextPOSIXProcessMonitor_mips64.cpp
  RegisterContextPOSIXProcessMonitor_powerpc.cpp
  RegisterContextPOSIXProcessMonitor_x86.cpp
  )
