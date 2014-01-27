set(LLVM_NO_RTTI 1)

add_lldb_library(lldbHostCommon
  Condition.cpp
  DynamicLibrary.cpp
  Editline.cpp
  File.cpp
  FileSpec.cpp
  Host.cpp
  Mutex.cpp
  OptionParser.cpp
  ProcessRunLock.cpp
  SocketAddress.cpp
  Symbols.cpp
  Terminal.cpp
  TimeValue.cpp
  )
