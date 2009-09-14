# See docs/CMake.html for instructions about how to build Compiler-RT with CMake.

PROJECT( CompilerRT C )
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

set(PACKAGE_NAME compiler-rt)
set(PACKAGE_VERSION 1.0svn)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

SET( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules )

# add definitions
include(DefineCompilerFlags)

# Disallow in-source build
INCLUDE( MacroEnsureOutOfSourceBuild )
MACRO_ENSURE_OUT_OF_SOURCE_BUILD(
 "${PROJECT_NAME} requires an out of source build. Please create a separate build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
 )

INCLUDE( ConfigureChecks.cmake )
CONFIGURE_FILE( config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h )

INCLUDE_DIRECTORIES(
 ${CMAKE_CURRENT_BINARY_DIR}
)

install(DIRECTORY include
  DESTINATION .
  PATTERN ".svn" EXCLUDE
  PATTERN "*.cmake" EXCLUDE
  PATTERN "*.in" EXCLUDE
  )

SET( Achitectures
 i386 x86_64 ppc arm
 )

SET( Configurations
 Debug Release Profile
 )

# BlocksRuntime - some random cruft that got thrown in at the last minute, ignoring for now.
ADD_SUBDIRECTORY( BlocksRuntime )

ADD_SUBDIRECTORY( lib )

# Tests are being ignored for until the very basics are working.
#INCLUDE( MacroAddCheckTest )
#ADD_SUBDIRECTORY( test )
