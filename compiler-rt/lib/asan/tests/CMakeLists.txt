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

include_directories(..)
include_directories(../..)

set(ASAN_UNITTEST_COMMON_CFLAGS
  -Wall
  -Wno-format
  -Werror
  -fvisibility=hidden
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

# Support 64-bit and 32-bit builds.
if(LLVM_BUILD_32_BITS)
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS -m32)
else()
  list(APPEND ASAN_UNITTEST_COMMON_CFLAGS -m64)
endif()

set(ASAN_GTEST_INCLUDE_CFLAGS
  -I${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include
  -I${LLVM_MAIN_SRC_DIR}/include
  -I${LLVM_BINARY_DIR}/include
  -D__STDC_CONSTANT_MACROS
  -D__STDC_LIMIT_MACROS
)

set(ASAN_UNITTEST_INSTRUMENTED_CFLAGS
  ${ASAN_UNITTEST_COMMON_CFLAGS}
  ${ASAN_GTEST_INCLUDE_CFLAGS}
  -faddress-sanitizer
  -mllvm "-asan-blacklist=${CMAKE_CURRENT_SOURCE_DIR}/asan_test.ignore"
  -mllvm -asan-stack=1
  -mllvm -asan-globals=1
  -mllvm -asan-mapping-scale=0        # default will be used
  -mllvm -asan-mapping-offset-log=-1  # default will be used
  -mllvm -asan-use-after-return=0
)

function(add_asan_test testsuite testname)
  add_unittest(${testsuite} ${testname} ${ARGN})
  if (APPLE)
    # Darwin-specific linker flags.
    set_property(TARGET ${testname} APPEND PROPERTY
                 LINK_FLAGS "-framework Foundation")
    target_link_libraries(${testname} clang_rt.asan_osx)
  elseif (ANDROID)
    target_link_libraries(${testname} clang_rt.asan-arm-android)
  elseif (UNIX)
    # Linux-specific linker flags.
    set_property(TARGET ${testname} APPEND PROPERTY
                 LINK_FLAGS "-lpthread -ldl -rdynamic")
    if(LLVM_BUILD_32_BITS)
      target_link_libraries(${testname} clang_rt.asan-i386)
    else()
      target_link_libraries(${testname} clang_rt.asan-x86_64)
    endif()
  endif()
  set(add_compile_flags "")
  get_property(compile_flags TARGET ${testname} PROPERTY COMPILE_FLAGS)
  foreach(arg ${ASAN_UNITTEST_COMMON_CFLAGS})
    set(add_compile_flags "${add_compile_flags} ${arg}")
  endforeach(arg ${ASAN_UNITTEST_COMMON_CFLAGS})
  set_property(TARGET ${testname} PROPERTY COMPILE_FLAGS
               "${compile_flags} ${add_compile_flags}")
endfunction()

set(ASAN_NOINST_TEST_SOURCES
  asan_noinst_test.cc
  asan_break_optimization.cc
  asan_test_main.cc
)

set(ASAN_INST_TEST_OBJECTS)

# We only support building instrumented tests when we're not cross compiling
# and targeting a unix-like system where we can predict viable compilation and
# linking strategies.
# We use a different approach to build these tests for Android. See below.
if("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND UNIX AND NOT ANDROID)

  # This function is a custom routine to manage manually compiling source files
  # for unit tests with the just-built Clang binary, using the ASan
  # instrumentation, and linking them into a test executable.
  function(add_asan_compile_command source extra_cflags)
    set(output_obj "${source}.asan.o")
    add_custom_command(
      OUTPUT ${output_obj}
      COMMAND clang
              ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS}
              ${extra_cflags}
              -c -o "${output_obj}"
              ${CMAKE_CURRENT_SOURCE_DIR}/${source}
      MAIN_DEPENDENCY ${source}
      DEPENDS clang ${ASAN_RUNTIME_LIBRARIES} ${ARGN}
      )
  endfunction()

  add_asan_compile_command(asan_globals_test.cc "")
  add_asan_compile_command(asan_test.cc "")
  list(APPEND ASAN_INST_TEST_OBJECTS asan_globals_test.cc.asan.o
                                     asan_test.cc.asan.o)
  if (APPLE)
    add_asan_compile_command(asan_mac_test.mm "-ObjC")
    list(APPEND ASAN_INST_TEST_OBJECTS asan_mac_test.mm.asan.o)
  endif()

  # Build benchmarks test instrumented with AddressSanitizer.
  add_asan_compile_command(asan_benchmarks_test.cc "")
  add_custom_target(AsanBenchmarks)
  set_target_properties(AsanBenchmarks PROPERTIES FOLDER "Asan benchmarks")
  add_asan_test(AsanBenchmarks AsanBenchmark asan_break_optimization.cc
                                             asan_benchmarks_test.cc.asan.o)
endif()

# Main AddressSanitizer unit tests.
add_custom_target(AsanUnitTests)
set_target_properties(AsanUnitTests PROPERTIES FOLDER "ASan unit tests")

if(ANDROID)
  set(ASAN_INST_TEST_SOURCES asan_globals_test.cc asan_test.cc)
  add_library(asan_noinst_test OBJECT
    ${ASAN_NOINST_TEST_SOURCES}
    )
  set_target_compile_flags(asan_noinst_test
    ${ASAN_UNITTEST_COMMON_CFLAGS} ${ASAN_GTEST_INCLUDE_CFLAGS}
    )
  add_asan_test(AsanUnitTests AsanTest
    ${ASAN_INST_TEST_SOURCES}
    $<TARGET_OBJECTS:asan_noinst_test>
    )
  set_target_compile_flags(AsanTest
    ${ASAN_UNITTEST_INSTRUMENTED_CFLAGS} ${ASAN_GTEST_INCLUDE_CFLAGS}
    )
  set_target_link_flags(AsanTest
    -pie
    )
else()
  add_asan_test(AsanUnitTests AsanTest ${ASAN_NOINST_TEST_SOURCES}
    ${ASAN_INST_TEST_OBJECTS})
endif()
