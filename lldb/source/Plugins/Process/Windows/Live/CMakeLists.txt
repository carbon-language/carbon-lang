include_directories(.)
include_directories(../../Utility)
include_directories(../Common)

set(PROC_WINDOWS_SOURCES
  DebuggerThread.cpp
  LocalDebugDelegate.cpp
  ProcessWindowsLive.cpp
  TargetThreadWindowsLive.cpp
  )

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(PROC_WINDOWS_SOURCES ${PROC_WINDOWS_SOURCES}
    x86/RegisterContextWindowsLive_x86.cpp
    )
elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(PROC_WINDOWS_SOURCES ${PROC_WINDOWS_SOURCES}
    x64/RegisterContextWindowsLive_x64.cpp
    )
endif()

add_lldb_library(lldbPluginProcessWindows
  ${PROC_WINDOWS_SOURCES}
  )
