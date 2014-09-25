set(LLVM_NO_RTTI 1)

include_directories(.)
include_directories(../POSIX)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessLinux
  LinuxThread.cpp
  NativeProcessLinux.cpp
  NativeRegisterContextLinux_x86_64.cpp
  NativeThreadLinux.cpp
  ProcessLinux.cpp
  ProcessMonitor.cpp
  ProcFileReader.cpp
  ThreadStateCoordinator.cpp
  )

