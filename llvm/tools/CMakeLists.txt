# NOTE: The tools are organized into five groups of four consisting of one
# large and three small executables. This is done to minimize memory load
# in parallel builds.  Please retain this ordering.

if( NOT WIN32 OR MSYS OR CYGWIN )
  # It is useful to build llvm-config before the other tools, so we
  # have a fresh LibDeps.txt for regenerating the hard-coded library
  # dependencies. llvm-config/CMakeLists.txt takes care of this but we
  # must keep llvm-config as the first entry on the list of tools to
  # be built.
  add_subdirectory(llvm-config)
endif()

add_subdirectory(opt)
add_subdirectory(llvm-as)
add_subdirectory(llvm-dis)
add_subdirectory(llvm-mc)

add_subdirectory(llc)
add_subdirectory(llvm-ranlib)
add_subdirectory(llvm-ar)
add_subdirectory(llvm-nm)

add_subdirectory(llvm-ld)
add_subdirectory(llvm-prof)
add_subdirectory(llvm-link)
add_subdirectory(lli)

add_subdirectory(llvm-extract)

add_subdirectory(bugpoint)
add_subdirectory(bugpoint-passes)
add_subdirectory(llvm-bcanalyzer)
add_subdirectory(llvm-stub)
add_subdirectory(edis)
add_subdirectory(llvmc)

if( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/clang/CMakeLists.txt )
  add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/clang )
endif( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/clang/CMakeLists.txt )

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} PARENT_SCOPE)
