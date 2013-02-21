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
  RegisterContext_i386.cpp
  RegisterContext_x86_64.cpp
  )
