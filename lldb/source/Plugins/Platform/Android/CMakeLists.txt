set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginPlatformAndroid
  AdbClient.cpp
  PlatformAndroid.cpp
  PlatformAndroidRemoteGDBServer.cpp
  )
