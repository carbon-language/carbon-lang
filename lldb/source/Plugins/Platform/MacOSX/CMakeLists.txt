list(APPEND PLUGIN_PLATFORM_MACOSX_SOURCES
  PlatformDarwin.cpp
  PlatformDarwinKernel.cpp
  PlatformMacOSX.cpp
  PlatformRemoteiOS.cpp
  )

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
list(APPEND PLUGIN_PLATFORM_MACOSX_SOURCES
  PlatformAppleSimulator.cpp
  PlatformiOSSimulatorCoreSimulatorSupport.mm
  PlatformiOSSimulator.cpp
  )
endif()

add_lldb_library(lldbPluginPlatformMacOSX ${PLUGIN_PLATFORM_MACOSX_SOURCES})
