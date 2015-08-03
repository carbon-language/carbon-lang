include_directories(../Utility)

add_lldb_library(lldbPluginProcessWinMiniDump
  ProcessWinMiniDump.cpp
  RegisterContextWindowsMiniDump.cpp
  ThreadWinMiniDump.cpp
  )
