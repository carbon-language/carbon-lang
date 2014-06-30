set(LLVM_NO_RTTI 1)

include_directories(.)
include_directories(../POSIX)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessLinux
  NativeProcessLinux.cpp
  NativeThreadLinux.cpp
  ProcessLinux.cpp
  ProcessMonitor.cpp
  ProcFileReader.cpp
  LinuxThread.cpp
  )

