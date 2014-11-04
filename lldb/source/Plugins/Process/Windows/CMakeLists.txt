set(LLVM_NO_RTTI 1)

include_directories(.)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessWindows
  DebugMonitorMessages.cpp
  DebugMonitorMessageResults.cpp
  DebugOneProcessThread.cpp
  DebugProcessLauncher.cpp
  DebugDriverThread.cpp
  ProcessWindows.cpp
  )

