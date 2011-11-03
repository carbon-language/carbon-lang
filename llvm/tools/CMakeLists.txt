# NOTE: The tools are organized into groups of four consisting of one large and
# three small executables. This is done to minimize memory load in parallel
# builds.  Please retain this ordering.

# If polly exists and is not disabled compile it and add it to the LLVM tools.
option(LLVM_BUILD_POLLY "Compile polly" ON)
if( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/polly/CMakeLists.txt )
  if (LLVM_BUILD_POLLY)
    add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/polly)
  endif (LLVM_BUILD_POLLY)
endif( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/polly/CMakeLists.txt )

if( NOT WIN32 OR MSYS OR CYGWIN )
  # It is useful to build llvm-config before the other tools, so we
  # have a fresh LibDeps.txt for regenerating the hard-coded library
  # dependencies. llvm-config/CMakeLists.txt takes care of this but we
  # must keep llvm-config as the first entry on the list of tools to
  # be built.
  add_subdirectory(llvm-config)

  # We currently require 'sed' to build llvm-config-2, so don't try to build it
  # on pure Win32.
  add_subdirectory(llvm-config-2)
endif()

add_subdirectory(opt)
add_subdirectory(llvm-as)
add_subdirectory(llvm-dis)
add_subdirectory(llvm-mc)

add_subdirectory(llc)
add_subdirectory(llvm-ranlib)
add_subdirectory(llvm-ar)
add_subdirectory(llvm-nm)
add_subdirectory(llvm-size)

add_subdirectory(llvm-ld)
add_subdirectory(llvm-prof)
add_subdirectory(llvm-link)
add_subdirectory(lli)

add_subdirectory(llvm-extract)
add_subdirectory(llvm-diff)
add_subdirectory(macho-dump)
add_subdirectory(llvm-objdump)
add_subdirectory(llvm-rtdyld)
add_subdirectory(llvm-dwarfdump)

add_subdirectory(bugpoint)
add_subdirectory(bugpoint-passes)
add_subdirectory(llvm-bcanalyzer)
add_subdirectory(llvm-stub)
add_subdirectory(edis)

if( NOT WIN32 )
  add_subdirectory(lto)
endif()

if( LLVM_ENABLE_PIC )
  # TODO: support other systems:
  if( CMAKE_SYSTEM_NAME STREQUAL "Linux" )
    add_subdirectory(gold)
  endif()
endif()

set(LLVM_CLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/clang" CACHE PATH "Path to Clang source directory")

if (NOT ${LLVM_CLANG_SOURCE_DIR} STREQUAL ""
    AND EXISTS ${LLVM_CLANG_SOURCE_DIR}/CMakeLists.txt)
  option(LLVM_BUILD_CLANG "Whether to build Clang as part of LLVM" ON)
  if (${LLVM_BUILD_CLANG})
    add_subdirectory(${LLVM_CLANG_SOURCE_DIR} clang)
  endif()
endif ()

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} PARENT_SCOPE)
