set(LLVM_NO_RTTI 1)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)
include_directories(${LLDB_SOURCE_DIR}/source)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  include_directories(MacOSX)
  #include_directories(${CMAKE_CURRENT_BINARY_DIR}/MacOSX)

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -Wl,-sectcreate,__TEXT,__info_plist,${CMAKE_CURRENT_SOURCE_DIR}/../resources/lldb-debugserver-Info.plist")
endif()

add_definitions(
  -DDEBUGSERVER_VERSION_STR="${LLDB_VERSION}"
  )

add_library(lldbDebugserverCommon
  #${dnbconfig_header}
  debugserver.cpp
  DNBArch.cpp
  DNBBreakpoint.cpp
  DNB.cpp
  DNBDataRef.cpp
  DNBError.cpp
  DNBLog.cpp
  DNBRegisterInfo.cpp
  DNBThreadResumeActions.cpp
  libdebugserver.cpp
  PseudoTerminal.cpp
  PThreadEvent.cpp
  PThreadMutex.cpp
  RNBContext.cpp
  RNBRemote.cpp
  RNBServices.cpp
  RNBSocket.cpp
  SysSignal.cpp
  TTYState.cpp
  )

add_dependencies(lldbDebugserverCommon generate_dnbconfig)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  find_library(COCOA_LIBRARY Cocoa)
  target_link_libraries(lldbDebugserverCommon ${COCOA_LIBRARY})
  add_subdirectory(MacOSX)
endif()

