# See docs/CMake.html for instructions about how to build Compiler-RT with CMake.

project(CompilerRT)
cmake_minimum_required(VERSION 2.6.1)

set(PACKAGE_NAME compiler-rt)
set(PACKAGE_VERSION 1.0svn)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed.
CMake would overwrite the makefiles distributed with Compiler-RT.
Please create a directory and run cmake from there, passing the path
to this source directory as the last argument.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
Please delete them.")
endif()

install(DIRECTORY include
  DESTINATION .
  PATTERN ".svn" EXCLUDE
  PATTERN "*.cmake" EXCLUDE
  PATTERN "*.in" EXCLUDE
  )

# BlocksRuntime - some random cruft that got thrown in at the last minute, ignoring for now.
# ADD_SUBDIRECTORY( BlocksRuntime )
ADD_SUBDIRECTORY( lib )
# Tests are being ignored for until the very basics are working.
# ADD_SUBDIRECTORY( test )

# Creates a shared lib .so
ADD_LIBRARY( ${PROJECT_NAME} SHARED ${SRCS})
