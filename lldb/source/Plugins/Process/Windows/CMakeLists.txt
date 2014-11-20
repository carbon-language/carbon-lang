set(LLVM_NO_RTTI 1)

include_directories(.)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessWindows
  DebuggerThread.cpp
  LocalDebugDelegate.cpp
  ProcessWindows.cpp
  RegisterContextWindows_x86.cpp
  TargetThreadWindows.cpp
  )

