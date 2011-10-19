########################################################################
# Experimental CMake build script for Google Test.
#
# Consider this a prototype.  It will change drastically.  For now,
# this is only for people on the cutting edge.
#
# To run the tests for Google Test itself on Linux, use 'make test' or
# ctest.  You can select which tests to run using 'ctest -R regex'.
# For more options, run 'ctest --help'.
########################################################################
#
# Project-wide settings

# Where gtest's .h files can be found.
include_directories(
  googletest/include
  )

if(WIN32)
  add_definitions(-DGTEST_OS_WINDOWS=1)
endif()

if(SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  add_definitions("-Wno-variadic-macros")
endif()

set(LLVM_REQUIRES_RTTI 1)
add_definitions( -DGTEST_HAS_RTTI=0 )

add_llvm_library(gtest
  googletest/gtest.cc
  googletest/gtest-death-test.cc
  googletest/gtest-filepath.cc
  googletest/gtest-port.cc
  googletest/gtest-printers.cc
  googletest/gtest-test-part.cc
  googletest/gtest-typed-test.cc
  )

add_llvm_library(gtest_main
  UnitTestMain/TestMain.cpp
  )

add_llvm_library_dependencies(gtest
  LLVMSupport
  )

add_llvm_library_dependencies(gtest_main
  gtest
  )
