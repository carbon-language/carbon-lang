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

# Need to export the API in the liblldb.dll for Windows
# The lldbAPI source files are added directly in liblldb
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows" )
  add_subdirectory(API)
endif ()

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

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
add_definitions( -DEXPORT_LIBLLDB )
endif()

# Need to export the API in the liblldb.dll for Windows
# The lldbAPI source files are added directly in liblldb
if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
# On Non-Windows, the corresponding file list is maintained in
# source\API\CMakeLists.txt.  When editing this list, do not forget to make a
# corresponding change in that file as well (when appropriate).
add_lldb_library(liblldb SHARED
  lldb.cpp
  lldb-log.cpp
  API/SBAddress.cpp
  API/SBBlock.cpp
  API/SBBreakpoint.cpp
  API/SBBreakpointLocation.cpp
  API/SBBroadcaster.cpp
  API/SBCommandInterpreter.cpp
  API/SBCommandReturnObject.cpp
  API/SBCommunication.cpp
  API/SBCompileUnit.cpp
  API/SBData.cpp
  API/SBDebugger.cpp
  API/SBDeclaration.cpp
  API/SBError.cpp
  API/SBEvent.cpp
  API/SBExecutionContext.cpp
  API/SBExpressionOptions.cpp
  API/SBFileSpec.cpp
  API/SBFileSpecList.cpp
  API/SBFrame.cpp
  API/SBFunction.cpp
  API/SBHostOS.cpp
  API/SBInstruction.cpp
  API/SBInstructionList.cpp
  API/SBLineEntry.cpp
  API/SBListener.cpp
  API/SBModule.cpp
  API/SBModuleSpec.cpp
  API/SBPlatform.cpp
  API/SBProcess.cpp
  API/SBQueue.cpp
  API/SBQueueItem.cpp
  API/SBSection.cpp
  API/SBSourceManager.cpp
  API/SBStream.cpp
  API/SBStringList.cpp
  API/SBSymbol.cpp
  API/SBSymbolContext.cpp
  API/SBSymbolContextList.cpp
  API/SBTarget.cpp
  API/SBThread.cpp
  API/SBThreadPlan.cpp
  API/SBThreadCollection.cpp
  API/SBType.cpp
  API/SBTypeCategory.cpp
  API/SBTypeEnumMember.cpp
  API/SBTypeFilter.cpp
  API/SBTypeFormat.cpp
  API/SBTypeNameSpecifier.cpp
  API/SBTypeSummary.cpp
  API/SBTypeSynthetic.cpp
  API/SBUnixSignals.cpp
  API/SBValue.cpp
  API/SBValueList.cpp
  API/SBWatchpoint.cpp
  Host/Windows/getopt/GetOptInc.cpp
  ${LLDB_WRAP_PYTHON}
  ${LLDB_VERS_GENERATED_FILE}
  )
else ()
add_lldb_library(liblldb SHARED
  lldb.cpp
  lldb-log.cpp
  ${LLDB_WRAP_PYTHON}
  ${LLDB_VERS_GENERATED_FILE}
  )
endif ()

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
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

if ( LLDB_ENABLE_PYTHON_SCRIPTS_SWIG_API_GENERATION )
	# Add a Post-Build Event to copy over Python files and create the symlink to liblldb.so for the Python API(hardlink on Windows)
	if ( NOT LLDB_DISABLE_PYTHON )
	    add_custom_command( TARGET liblldb
	        POST_BUILD
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/finishSwigWrapperClasses.py
	        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/finishSwigWrapperClasses.py --buildConfig=${CMAKE_BUILD_TYPE} "--srcRoot=${LLDB_SOURCE_DIR}" "--targetDir=${CMAKE_CURRENT_BINARY_DIR}/../scripts" "--cfgBldDir=${CMAKE_CURRENT_BINARY_DIR}/../scripts" "--prefix=${CMAKE_BINARY_DIR}" "--cmakeBuildConfiguration=${CMAKE_CFG_INTDIR}" -m
	        COMMENT "Python script sym-linking LLDB Python API")
	endif ()
endif ()

install(TARGETS liblldb
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
