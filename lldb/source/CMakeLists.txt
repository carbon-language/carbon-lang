include_directories(.)

if (__ANDROID_NDK__ OR (CMAKE_SYSTEM_NAME MATCHES "Windows"))
  set(LLDB_DEFAULT_DISABLE_LIBEDIT 1)
else()
  set(LLDB_DEFAULT_DISABLE_LIBEDIT 0)
endif ()

set(LLDB_DISABLE_LIBEDIT ${LLDB_DEFAULT_DISABLE_LIBEDIT} CACHE BOOL "Disables the use of editline.")
if (LLDB_DISABLE_LIBEDIT)
  add_definitions( -DLLDB_DISABLE_LIBEDIT )
endif()

if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
include_directories(
  Plugins/Process/Linux
  Plugins/Process/POSIX
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
include_directories(
  Plugins/Process/FreeBSD
  Plugins/Process/POSIX
  )
endif ()

add_subdirectory(API)
add_subdirectory(Breakpoint)
add_subdirectory(Commands)
add_subdirectory(Core)
add_subdirectory(DataFormatters)
add_subdirectory(Expression)
add_subdirectory(Host)
add_subdirectory(Interpreter)
add_subdirectory(Plugins)
add_subdirectory(Symbol)
add_subdirectory(Target)
add_subdirectory(Utility)

include(../cmake/LLDBDependencies.cmake)

add_lldb_library(liblldb SHARED
  lldb.cpp
  lldb-log.cpp
  ${LLDB_WRAP_PYTHON}
  ${LLDB_VERS_GENERATED_FILE}
  )

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
target_link_libraries(liblldb PRIVATE lldbAPI)

set_target_properties(liblldb
  PROPERTIES
  OUTPUT_NAME liblldb
  VERSION ${LLDB_VERSION}
  )
else()
set_target_properties(liblldb
  PROPERTIES
  OUTPUT_NAME lldb
  VERSION ${LLDB_VERSION}
  )
endif()

if (LLDB_WRAP_PYTHON OR LLDB_VERS_GENERATED_FILE)
  add_dependencies(liblldb swig_wrapper)
  message("Adding dependency on swig_wrapper")
endif()
target_link_libraries(liblldb ${cmake_2_8_12_PRIVATE} ${LLDB_SYSTEM_LIBS})

# Determine LLDB revision and repository. GetSourceVersion and GetRepositoryPath are shell-scripts, and as
# such will not work on Windows.
if ( NOT CMAKE_SYSTEM_NAME MATCHES "Windows" )
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/utils/GetSourceVersion ${LLDB_SOURCE_DIR}
                  OUTPUT_VARIABLE LLDB_REVISION)
  if ( LLDB_REVISION )
    string(REGEX REPLACE "(\r?\n)+$" "" LLDB_REVISION ${LLDB_REVISION})
  endif()

  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/utils/GetRepositoryPath ${LLDB_SOURCE_DIR}
                  OUTPUT_VARIABLE LLDB_REPOSITORY)
  if ( LLDB_REPOSITORY )
    # Replace newline characters with spaces
    string(REGEX REPLACE "(\r?\n)+" " " LLDB_REPOSITORY ${LLDB_REPOSITORY})

    # Remove trailing spaces
    string(REGEX REPLACE "(\ )+$" "" LLDB_REPOSITORY ${LLDB_REPOSITORY})
  endif()

  set_property(
    SOURCE lldb.cpp
    PROPERTY COMPILE_DEFINITIONS "LLDB_REVISION=\"${LLDB_REVISION}\"" "LLDB_REPOSITORY=\"${LLDB_REPOSITORY}\"")
endif ()
# FIXME: implement svn/git revision and repository parsing solution on Windows. There is an SVN-only
#        revision parsing solution in tools/clang/lib/Basic/CMakelists.txt.

install(TARGETS liblldb
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
