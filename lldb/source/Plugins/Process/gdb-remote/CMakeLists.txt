set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginProcessGDBRemote
  GDBRemoteCommunication.cpp
  GDBRemoteCommunicationClient.cpp
  GDBRemoteCommunicationServer.cpp
  GDBRemoteCommunicationServerCommon.cpp
  GDBRemoteCommunicationServerLLGS.cpp
  GDBRemoteCommunicationServerPlatform.cpp
  GDBRemoteRegisterContext.cpp
  ProcessGDBRemote.cpp
  ProcessGDBRemoteLog.cpp
  ThreadGDBRemote.cpp
  )