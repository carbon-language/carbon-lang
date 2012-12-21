# Testing rules for AddressSanitizer.
#
# These are broken into two buckets. One set of tests directly interacts with
# the runtime library and checks its functionality. These are the
# no-instrumentation tests.
#
# Another group of tests relies upon the ability to compile the test with
# address sanitizer instrumentation pass. These tests form "integration" tests
# and have some elements of version skew -- they test the *host* compiler's
# instrumentation against the just-built runtime library.

include(CheckCXXCompilerFlag)
include(CompilerRTCompile)
include(CompilerRTUnittests)

include_directories(..)
include_directories(../..)

set(ASAN_UNITTEST_COMMON_CFLAGS
  ${COMPILER_RT_GTEST_INCLUDE_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/asan
  -Wall
  -Wno-format
  -Werror
  -g
  -O2
)

if(SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS -Wno-variadic-macros)
endif()

# Use -D instead of definitions to please custom compile command.
if(ANDROID)
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS
    -DASAN_LOW_MEMORY=1
    -DASAN_HAS_BLACKLIST=1
    -DASAN_HAS_EXCEPTIONS=1
    -DASAN_NEEDS_SEGV=0
    -DASAN_UAR=0
    -fPIE
  )
else()
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS
    -DASAN_HAS_BLACKLIST=1
    -DASAN_HAS_EXCEPTIONS=1
    -DASAN_NEEDS_SEGV=1
    -DASAN_UAR=0
  )
endif()

set(ASAN_LINK_FLAGS)
if(ANDROID)
  # On Android, we link with ASan runtime manually
  list(APPEND ASAN_LINK_FLAGS -pie)
else()
  # On other platforms, we depend on Clang driver behavior,
  # passing -fsanitize=address flag.
  list(APPEND ASAN_LINK_FLAGS -fsanitize=address)
endif()
# Unit tests on Mac depend on Foundation.
if(APPLE)
  list(APPEND ASAN_LINK_FLAGS -framework Foundation)
endif()
# Unit tests require libstdc++.
list(APPEND ASAN_LINK_FLAGS -lstdc++)

set(ASAN_BLACKLIST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/asan_test.ignore")

set(ASAN_UNITTEST_INSTRUMENTED_CFLAGS
  ${ASAN_UNITTEST_COMMON_CFLAGS}
  -fsanitize=address
  -mllvm "-asan-blacklist=${ASAN_BLACKLIST_FILE}"
  -mllvm -asan-stack=1
  -mllvm -asan-globals=1
  -mllvm -asan-mapping-scale=0        # default will be used
  -mllvm -asan-mapping-offset-log=-1  # default will be used
  -mllvm -asan-use-after-return=0
)

# Compile source for the given architecture, using compiler
# options in ${ARGN}, and add it to the object list.
macro(asan_compile obj_list source arch)
  get_filename_component(basename ${source} NAME)
  set(output_obj "${basename}.${arch}.o")
  get_target_flags_for_arch(${arch} TARGET_CFLAGS)
  clang_compile(${output_obj} ${source}
                CFLAGS ${ARGN} ${TARGET_CFLAGS}
                DEPS gtest ${ASAN_RUNTIME_LIBRARIES}
                           ${ASAN_BLACKLIST_FILE})
  list(APPEND ${obj_list} ${output_obj})
endmacro()

# Link ASan unit test for a given architecture from a set
# of objects in ${ARGN}.
macro(add_asan_test test_suite test_name arch)
  get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
  add_compiler_rt_test(${test_suite} ${test_name}
                       OBJECTS ${ARGN}
                       DEPS ${ASAN_RUNTIME_LIBRARIES} ${ARGN}
                       LINK_FLAGS ${ASAN_LINK_FLAGS}
                                  ${TARGET_LINK_FLAGS})
endmacro()

# Main AddressSanitizer unit tests.
add_custom_target(AsanUnitTests)
set_target_properties(AsanUnitTests PROPERTIES FOLDER "ASan unit tests")
# ASan benchmarks (not actively used now).
add_custom_target(AsanBenchmarks)
set_target_properties(AsanBenchmarks PROPERTIES FOLDER "Asan benchmarks")

# Adds ASan unit tests and benchmarks for architecture.
macro(add_asan_tests_for_arch arch)
  # Build gtest instrumented with ASan.
  set(ASAN_INST_GTEST)
  asan_compile(ASAN_INST_GTEST ${COMPILER_RT_GTEST_SOURCE} ${arch} 
                               ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS})
  # Instrumented tests.
  set(ASAN_INST_TEST_OBJECTS)
  asan_compile(ASAN_INST_TEST_OBJECTS asan_globals_test.cc ${arch}
               ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS})
  asan_compile(ASAN_INST_TEST_OBJECTS asan_test.cc ${arch}
               ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS})
  if (APPLE)
    asan_compile(ASAN_INST_TEST_OBJECTS asan_mac_test.mm ${arch}
                 ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS} -ObjC)
  endif()
  # Uninstrumented tests.
  set(ASAN_NOINST_TEST_OBJECTS)
  asan_compile(ASAN_NOINST_TEST_OBJECTS asan_noinst_test.cc ${arch}
               ${ASAN_UNITTEST_COMMON_CFLAGS})
  asan_compile(ASAN_NOINST_TEST_OBJECTS asan_test_main.cc ${arch}
               ${ASAN_UNITTEST_COMMON_CFLAGS})
  # Link everything together.
  add_asan_test(AsanUnitTests "Asan-${arch}-Test" ${arch}
                ${ASAN_NOINST_TEST_OBJECTS}
                ${ASAN_INST_TEST_OBJECTS} ${ASAN_INST_GTEST})

  # Instrumented benchmarks.
  set(ASAN_BENCHMARKS_OBJECTS)
  asan_compile(ASAN_BENCHMARKS_OBJECTS asan_benchmarks_test.cc ${arch}
               ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS})
  # Link benchmarks.
  add_asan_test(AsanBenchmarks "Asan-${arch}-Benchmark" ${arch}
                ${ASAN_BENCHMARKS_OBJECTS} ${ASAN_INST_GTEST})
endmacro()

# We only support building instrumented tests when we're not cross compiling
# and targeting a unix-like system where we can predict viable compilation and
# linking strategies.
# We use a different approach to build these tests for Android. See below.
if("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND UNIX AND NOT ANDROID)
  if(CAN_TARGET_X86_64)
    add_asan_tests_for_arch(x86_64)
  endif()
  if(CAN_TARGET_I386)
    add_asan_tests_for_arch(i386)
  endif()
endif()

if(ANDROID)
  # We assume that unit tests on Android are built in a build
  # tree with fresh Clang as a host compiler.
  set(ASAN_NOINST_TEST_SOURCES asan_noinst_test.cc asan_test_main.cc)
  set(ASAN_INST_TEST_SOURCES asan_globals_test.cc asan_test.cc)
  add_library(asan_noinst_test OBJECT ${ASAN_NOINST_TEST_SOURCES})
  set_target_compile_flags(asan_noinst_test ${ASAN_UNITTEST_COMMON_CFLAGS})
  add_library(asan_inst_test OBJECT
              ${ASAN_INST_TEST_SOURCES} ${COMPILER_RT_GTEST_SOURCE})  
  set_target_compile_flags(asan_inst_test ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS})
  add_executable(AsanTest
    $<TARGET_OBJECTS:asan_noinst_test>
    $<TARGET_OBJECTS:asan_inst_test>
  )
  # Setup correct output directory and link flags.
  get_unittest_directory(OUTPUT_DIR)
  set_target_properties(AsanTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
  set_target_link_flags(AsanTest ${ASAN_LINK_FLAGS})
  target_link_libraries(AsanTest clang_rt.asan-arm-android)
  # Add unit test to test suite.
  add_dependencies(AsanUnitTests AsanTest)
endif()
