include_directories(../Utility)

set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginProcessElfCore
  ProcessElfCore.cpp
  ThreadElfCore.cpp
  RegisterContextPOSIXCore_mips64.cpp
  RegisterContextPOSIXCore_x86_64.cpp
  )
