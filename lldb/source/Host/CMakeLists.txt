macro(add_host_subdirectory group)
  list(APPEND HOST_SOURCES ${ARGN})
  source_group(${group} FILES ${ARGN})
endmacro()

add_host_subdirectory(common
  common/Condition.cpp
  common/File.cpp
  common/FileCache.cpp
  common/FileSpec.cpp
  common/Host.cpp
  common/HostInfoBase.cpp
  common/HostNativeThreadBase.cpp
  common/HostProcess.cpp
  common/HostThread.cpp
  common/IOObject.cpp
  common/Mutex.cpp
  common/MonitoringProcessLauncher.cpp
  common/NativeBreakpoint.cpp
  common/NativeBreakpointList.cpp
  common/NativeProcessProtocol.cpp
  common/NativeThreadProtocol.cpp
  common/OptionParser.cpp
  common/ProcessRunLock.cpp
  common/Socket.cpp
  common/SocketAddress.cpp
  common/SoftwareBreakpoint.cpp
  common/Symbols.cpp
  common/Terminal.cpp
  common/ThisThread.cpp
  common/ThreadLauncher.cpp
  common/TimeValue.cpp
  )

if (NOT LLDB_DISABLE_LIBEDIT)
  add_host_subdirectory(common
    common/Editline.cpp
    )
endif()

add_host_subdirectory(posix
  posix/ConnectionFileDescriptorPosix.cpp
  )

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  add_host_subdirectory(windows
    windows/Condition.cpp
    windows/ConnectionGenericFileWindows.cpp
    windows/EditLineWin.cpp
    windows/FileSystem.cpp
    windows/Host.cpp
    windows/HostInfoWindows.cpp
    windows/HostProcessWindows.cpp
    windows/HostThreadWindows.cpp
    windows/Mutex.cpp
    windows/PipeWindows.cpp
    windows/ProcessLauncherWindows.cpp
    windows/ProcessRunLock.cpp
    windows/ThisThread.cpp
    windows/Windows.cpp
    )
else()
  add_host_subdirectory(posix
    posix/FileSystem.cpp
    posix/HostInfoPosix.cpp
    posix/HostProcessPosix.cpp
    posix/HostThreadPosix.cpp
    posix/PipePosix.cpp
    posix/ProcessLauncherPosix.cpp
    )

  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    include_directories(SYSTEM ${LIBXML2_INCLUDE_DIR})
    add_host_subdirectory(macosx
      macosx/Host.mm
      macosx/HostInfoMacOSX.mm
      macosx/HostThreadMacOSX.mm
      macosx/Symbols.cpp
      macosx/ThisThread.cpp
      macosx/cfcpp/CFCBundle.cpp
      macosx/cfcpp/CFCData.cpp
      macosx/cfcpp/CFCMutableArray.cpp
      macosx/cfcpp/CFCMutableDictionary.cpp
      macosx/cfcpp/CFCMutableSet.cpp
      macosx/cfcpp/CFCString.cpp
      )

  elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    if (__ANDROID_NDK__)
      add_host_subdirectory(android
        linux/Host.cpp
        linux/HostInfoLinux.cpp
        linux/HostThreadLinux.cpp
        linux/ThisThread.cpp
        )
    else()
      add_host_subdirectory(linux
        linux/Host.cpp
        linux/HostInfoLinux.cpp
        linux/HostThreadLinux.cpp
        linux/ThisThread.cpp
        )
    endif()
  elseif (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_host_subdirectory(freebsd
      freebsd/Host.cpp
      freebsd/HostInfoFreeBSD.cpp
      freebsd/HostThreadFreeBSD.cpp
      freebsd/ThisThread.cpp
      )
  endif()
endif()

add_lldb_library(lldbHost ${HOST_SOURCES})
