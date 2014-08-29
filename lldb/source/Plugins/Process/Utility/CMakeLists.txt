include_directories(../../../Utility/)

set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginProcessUtility
  DynamicRegisterInfo.cpp
  FreeBSDSignals.cpp
  HistoryThread.cpp
  HistoryUnwind.cpp
  InferiorCallPOSIX.cpp
  LinuxSignals.cpp
  RegisterContextDarwin_arm.cpp
  RegisterContextDarwin_arm64.cpp
  RegisterContextDarwin_i386.cpp
  RegisterContextDarwin_x86_64.cpp
  RegisterContextDummy.cpp
  RegisterContextFreeBSD_i386.cpp
  RegisterContextFreeBSD_mips64.cpp
  RegisterContextFreeBSD_x86_64.cpp
  RegisterContextHistory.cpp
  RegisterContextLinux_arm64.cpp
  RegisterContextLinux_i386.cpp
  RegisterContextLinux_x86_64.cpp
  RegisterContextLLDB.cpp
  RegisterContextMacOSXFrameBackchain.cpp
  RegisterContextMach_arm.cpp
  RegisterContextMach_i386.cpp
  RegisterContextMach_x86_64.cpp
  RegisterContextMemory.cpp
  RegisterContextPOSIX_mips64.cpp
  RegisterContextPOSIX_x86.cpp
  RegisterContextThreadMemory.cpp
  StopInfoMachException.cpp
  ThreadMemory.cpp
  UnwindLLDB.cpp
  UnwindMacOSXFrameBackchain.cpp
  )
