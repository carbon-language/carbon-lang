set(LLVM_NO_RTTI 1)

macro(add_process_windows_subdirectory group)
  list(APPEND PROC_WINDOWS_SOURCES ${ARGN})
  source_group(${group} FILES ${ARGN})
endmacro()

include_directories(.)
include_directories(../Utility)

add_process_windows_subdirectory(common
  DebuggerThread.cpp
  DynamicLoaderWindows.cpp
  LocalDebugDelegate.cpp
  ProcessWindows.cpp
  TargetThreadWindows.cpp
  )

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
  add_process_windows_subdirectory(x86
  x86/RegisterContextWindows_x86.cpp
  )
endif()

add_lldb_library(lldbPluginProcessWindows
  ${PROC_WINDOWS_SOURCES}
  )
