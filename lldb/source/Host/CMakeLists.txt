macro(add_host_subdirectory group)
  list(APPEND HOST_SOURCES ${ARGN})
  source_group(group FILES ${ARGN})
endmacro()

add_host_subdirectory(common
  common/Condition.cpp
  common/DynamicLibrary.cpp
  common/Editline.cpp
  common/File.cpp
  common/FileCache.cpp
  common/FileSpec.cpp
  common/Host.cpp
  common/IOObject.cpp
  common/Mutex.cpp
  common/NativeBreakpoint.cpp
  common/NativeBreakpointList.cpp
  common/NativeProcessProtocol.cpp
  common/NativeThreadProtocol.cpp
  common/OptionParser.cpp
  common/Pipe.cpp
  common/ProcessRunLock.cpp
  common/Socket.cpp
  common/SocketAddress.cpp
  common/SoftwareBreakpoint.cpp
  common/Symbols.cpp
  common/Terminal.cpp
  common/TimeValue.cpp
  )

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_host_subdirectory(windows
    windows/FileSystem.cpp
    windows/Host.cpp
    windows/ProcessRunLock.cpp
    windows/Mutex.cpp
    windows/Condition.cpp
    windows/Windows.cpp
    windows/EditLineWin.cpp
    )
else()
  add_host_subdirectory(posix
    posix/FileSystem.cpp
    )

  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    include_directories(SYSTEM ${LIBXML2_INCLUDE_DIR})
    add_host_subdirectory(macosx
      macosx/Host.mm
      macosx/Symbols.cpp
      macosx/cfcpp/CFCBundle.cpp
      macosx/cfcpp/CFCData.cpp
      macosx/cfcpp/CFCMutableArray.cpp
      macosx/cfcpp/CFCMutableDictionary.cpp
      macosx/cfcpp/CFCMutableSet.cpp
      macosx/cfcpp/CFCString.cpp
      )

  elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_host_subdirectory(linux
      linux/Host.cpp
      )
  elseif (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_host_subdirectory(freebsd
      freebsd/Host.cpp
      )
  endif()
endif()

add_lldb_library(lldbHost ${HOST_SOURCES})
