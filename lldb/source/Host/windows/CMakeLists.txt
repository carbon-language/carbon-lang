set(LLVM_NO_RTTI 1)

add_lldb_library(lldbHostWindows
	FileSystem.cpp
  Host.cpp
  ProcessRunLock.cpp
  Mutex.cpp
  Condition.cpp
  Windows.cpp
  EditLineWin.cpp
  )
