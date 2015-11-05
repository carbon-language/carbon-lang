list(APPEND PLUGIN_PLATFORM_MACOSX_SOURCES
  PlatformAppleSimulator.cpp
  PlatformDarwin.cpp
  PlatformDarwinKernel.cpp
  PlatformiOSSimulator.cpp
  PlatformMacOSX.cpp
  PlatformRemoteiOS.cpp
  )

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
list(APPEND PLUGIN_PLATFORM_MACOSX_SOURCES
  PlatformiOSSimulatorCoreSimulatorSupport.mm
  )
endif()

add_lldb_library(lldbPluginPlatformMacOSX ${PLUGIN_PLATFORM_MACOSX_SOURCES})
