set(LLVM_NO_RTTI 1)

include_directories(.)
include_directories(../Utility)

set(PROC_WINDOWS_SOURCES
  DebuggerThread.cpp
  DynamicLoaderWindows.cpp
  LocalDebugDelegate.cpp
  ProcessWindows.cpp
  ProcessWindowsLog.cpp
  RegisterContextWindows.cpp
  TargetThreadWindows.cpp
  )

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(PROC_WINDOWS_SOURCES ${PROC_WINDOWS_SOURCES}
    x86/RegisterContextWindows_x86.cpp
    )
elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(PROC_WINDOWS_SOURCES ${PROC_WINDOWS_SOURCES}
    x64/RegisterContextWindows_x64.cpp
    )
endif()

add_lldb_library(lldbPluginProcessWindows
  ${PROC_WINDOWS_SOURCES}
  )
