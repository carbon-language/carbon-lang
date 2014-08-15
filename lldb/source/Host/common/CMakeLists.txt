set(LLVM_NO_RTTI 1)

add_lldb_library(lldbHostCommon
  Condition.cpp
  DynamicLibrary.cpp
  Editline.cpp
  File.cpp
	FileCache.cpp
  FileSpec.cpp
  Host.cpp
  IOObject.cpp
  Mutex.cpp
  NativeBreakpoint.cpp
  NativeBreakpointList.cpp
  NativeProcessProtocol.cpp
  NativeThreadProtocol.cpp
  OptionParser.cpp
  Pipe.cpp
  ProcessRunLock.cpp
  Socket.cpp
  SocketAddress.cpp
  SoftwareBreakpoint.cpp
  Symbols.cpp
  Terminal.cpp
  TimeValue.cpp
  )
