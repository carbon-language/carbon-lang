include_directories(../Utility)

add_lldb_library(lldbPluginProcessMachCore
  ProcessMachCore.cpp
  ThreadMachCore.cpp
  )
