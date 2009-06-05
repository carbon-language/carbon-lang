# Project that runs the Clang regression tests for a given glob pattern. 
#
# There are several CMake cache variabes that must be set for this
# project to work:
#
#   CLANG_TEST_RUNNER: The TestRunner.sh shell script, which is used to test
#   Clang.
#
#   CLANG_TEST_GLOB_PATTERNS: Set to a GLOB pattern to identify the kind of
#   tests, e.g., *.cpp for C++ tests.
#
#   LLVM_TOOLS_PATH: The directory where the Clang and LLVM tool
#   executables (such as opt) are generated.
#
#   LLVM_SCRIPTS_PATH: The directory where the LLVM test scripts are
#   located.
cmake_minimum_required(VERSION 2.6)
project(ClangTest)

enable_testing()

# Computes the normalized name of a test from its path name.
macro(compute_test_name var filename)
  get_filename_component(test_name ${filename} NAME_WE)
  get_filename_component(test_path ${filename} PATH)
  get_filename_component(test_lastpath ${test_path} NAME_WE)
  set(${var} "${test_lastpath}-${test_name}")
endmacro()

# FIXME: This is a hack to make sure that there is a proper "clean" target.
# There must be a better way.
file(WRITE dummy.c "int dummy() { return 0; }")
add_library(dummy dummy.c)
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "dummy.c")

# Set the path appropriately
set(PATH $ENV{PATH})
set(PATH "${LLVM_TOOLS_PATH}:${LLVM_SCRIPTS_PATH}:${PATH}")

# Find the test files
string(REPLACE ":" ";" CLANG_TEST_GLOB_PATTERNS ${CLANG_TEST_GLOB_PATTERNS})
separate_arguments(CLANG_TEST_GLOB_PATTERNS)
file(GLOB_RECURSE tests ${CLANG_TEST_GLOB_PATTERNS})

# Add individual tests for each file we found.
foreach(test ${tests})
  compute_test_name(testname ${test})
  add_test(${testname} ${CLANG_TEST_RUNNER} ${test})
  set_tests_properties(${testname} PROPERTIES
    ENVIRONMENT "PATH=${PATH}")
endforeach(test ${tests})