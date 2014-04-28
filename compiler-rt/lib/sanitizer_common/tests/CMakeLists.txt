include(CompilerRTCompile)

set(SANITIZER_UNITTESTS
  sanitizer_allocator_test.cc
  sanitizer_atomic_test.cc
  sanitizer_bitvector_test.cc
  sanitizer_bvgraph_test.cc
  sanitizer_common_test.cc
  sanitizer_deadlock_detector_test.cc
  sanitizer_flags_test.cc
  sanitizer_format_interceptor_test.cc
  sanitizer_ioctl_test.cc
  sanitizer_libc_test.cc
  sanitizer_linux_test.cc
  sanitizer_list_test.cc
  sanitizer_mutex_test.cc
  sanitizer_nolibc_test.cc
  sanitizer_posix_test.cc
  sanitizer_printf_test.cc
  sanitizer_procmaps_test.cc
  sanitizer_stackdepot_test.cc
  sanitizer_stacktrace_test.cc
  sanitizer_stoptheworld_test.cc
  sanitizer_suppressions_test.cc
  sanitizer_test_main.cc
  sanitizer_thread_registry_test.cc)

set(SANITIZER_TEST_HEADERS
  sanitizer_test_utils.h)
foreach(header ${SANITIZER_HEADERS})
  list(APPEND SANITIZER_TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/../${header})
endforeach()

set(SANITIZER_TEST_CFLAGS_COMMON
  ${COMPILER_RT_GTEST_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/include
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${COMPILER_RT_SOURCE_DIR}/lib/sanitizer_common
  -DGTEST_HAS_RTTI=0
  -O2 -g -fno-rtti
  -Werror=sign-compare
  -Wno-non-virtual-dtor)

set(SANITIZER_TEST_LINK_FLAGS_COMMON
  -lstdc++)
append_if(COMPILER_RT_HAS_LIBDL -ldl SANITIZER_TEST_LINK_FLAGS_COMMON)
append_if(COMPILER_RT_HAS_LIBPTHREAD -lpthread SANITIZER_TEST_LINK_FLAGS_COMMON)
# x86_64 FreeBSD 9.2 additionally requires libc++ to build the tests.
if(CMAKE_SYSTEM MATCHES "FreeBSD-9.2-RELEASE")
  list(APPEND SANITIZER_TEST_LINK_FLAGS_COMMON "-lc++")
endif()

include_directories(..)
include_directories(../..)

# Adds static library which contains sanitizer_common object file
# (universal binary on Mac and arch-specific object files on Linux).
macro(add_sanitizer_common_lib library)
  add_library(${library} STATIC ${ARGN})
  set_target_properties(${library} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endmacro()

function(get_sanitizer_common_lib_for_arch arch lib lib_name)
  if(APPLE)
    set(tgt_name "RTSanitizerCommon.test.osx")
  else()
    set(tgt_name "RTSanitizerCommon.test.${arch}")
  endif()
  set(${lib} "${tgt_name}" PARENT_SCOPE)
  set(${lib_name} "lib${tgt_name}.a" PARENT_SCOPE)
endfunction()

# Sanitizer_common unit tests testsuite.
add_custom_target(SanitizerUnitTests)
set_target_properties(SanitizerUnitTests PROPERTIES
  FOLDER "Sanitizer unittests")

# Adds sanitizer tests for architecture.
macro(add_sanitizer_tests_for_arch arch)
  get_target_flags_for_arch(${arch} TARGET_FLAGS)
  set(SANITIZER_TEST_SOURCES ${SANITIZER_UNITTESTS}
                             ${COMPILER_RT_GTEST_SOURCE})
  set(SANITIZER_TEST_COMPILE_DEPS ${SANITIZER_TEST_HEADERS})
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    list(APPEND SANITIZER_TEST_COMPILE_DEPS gtest)
  endif()
  set(SANITIZER_TEST_OBJECTS)
  foreach(source ${SANITIZER_TEST_SOURCES})
    get_filename_component(basename ${source} NAME)
    set(output_obj "${basename}.${arch}.o")
    clang_compile(${output_obj} ${source}
                  CFLAGS ${SANITIZER_TEST_CFLAGS_COMMON} ${TARGET_FLAGS}
                  DEPS ${SANITIZER_TEST_COMPILE_DEPS})
    list(APPEND SANITIZER_TEST_OBJECTS ${output_obj})
  endforeach()
  get_sanitizer_common_lib_for_arch(${arch} SANITIZER_COMMON_LIB
                                    SANITIZER_COMMON_LIB_NAME)
  # Add unittest target.
  set(SANITIZER_TEST_NAME "Sanitizer-${arch}-Test")
  add_compiler_rt_test(SanitizerUnitTests ${SANITIZER_TEST_NAME}
                       OBJECTS ${SANITIZER_TEST_OBJECTS}
                               ${SANITIZER_COMMON_LIB_NAME}
                       DEPS ${SANITIZER_TEST_OBJECTS} ${SANITIZER_COMMON_LIB}
                       LINK_FLAGS ${SANITIZER_TEST_LINK_FLAGS_COMMON}
                                  ${TARGET_FLAGS})

  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" AND "${arch}" STREQUAL "x86_64")
    # Test that the libc-independent part of sanitizer_common is indeed
    # independent of libc, by linking this binary without libc (here) and
    # executing it (unit test in sanitizer_nolibc_test.cc).
    clang_compile(sanitizer_nolibc_test_main.${arch}.o
                  sanitizer_nolibc_test_main.cc
                  CFLAGS ${SANITIZER_TEST_CFLAGS_COMMON} ${TARGET_FLAGS}
                  DEPS ${SANITIZER_TEST_COMPILE_DEPS})
    add_compiler_rt_test(SanitizerUnitTests "Sanitizer-${arch}-Test-Nolibc"
                         OBJECTS sanitizer_nolibc_test_main.${arch}.o
                                 -Wl,-whole-archive
                                 libRTSanitizerCommon.test.nolibc.${arch}.a
                                 -Wl,-no-whole-archive
                         DEPS sanitizer_nolibc_test_main.${arch}.o
                              RTSanitizerCommon.test.nolibc.${arch}
                         LINK_FLAGS -nostdlib ${TARGET_FLAGS})
  endif()
endmacro()

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  # We use just-built clang to build sanitizer_common unittests, so we must
  # be sure that produced binaries would work.
  if(APPLE)
    add_sanitizer_common_lib("RTSanitizerCommon.test.osx"
                             $<TARGET_OBJECTS:RTSanitizerCommon.osx>)
  else()
    if(CAN_TARGET_x86_64)
      add_sanitizer_common_lib("RTSanitizerCommon.test.x86_64"
                               $<TARGET_OBJECTS:RTSanitizerCommon.x86_64>
                               $<TARGET_OBJECTS:RTSanitizerCommonLibc.x86_64>)
      add_sanitizer_common_lib("RTSanitizerCommon.test.nolibc.x86_64"
                               $<TARGET_OBJECTS:RTSanitizerCommon.x86_64>)
    endif()
    if(CAN_TARGET_i386)
      add_sanitizer_common_lib("RTSanitizerCommon.test.i386"
                               $<TARGET_OBJECTS:RTSanitizerCommon.i386>
                               $<TARGET_OBJECTS:RTSanitizerCommonLibc.i386>)
    endif()
  endif()
  if(CAN_TARGET_x86_64)
    add_sanitizer_tests_for_arch(x86_64)
  endif()
  if(CAN_TARGET_i386)
    add_sanitizer_tests_for_arch(i386)
  endif()
endif()

if(ANDROID)
  # We assume that unit tests on Android are built in a build
  # tree with fresh Clang as a host compiler.
  add_executable(SanitizerTest
    ${SANITIZER_UNITTESTS}
    ${COMPILER_RT_GTEST_SOURCE}
    $<TARGET_OBJECTS:RTSanitizerCommon.arm.android>)
  set_target_compile_flags(SanitizerTest
    ${SANITIZER_COMMON_CFLAGS}
    ${SANITIZER_TEST_CFLAGS_COMMON})
  # Setup correct output directory and link flags.
  set_target_properties(SanitizerTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_target_link_flags(SanitizerTest ${SANITIZER_TEST_LINK_FLAGS_COMMON})
  target_link_libraries(SanitizerTest log)
  # Add unit test to test suite.
  add_dependencies(SanitizerUnitTests SanitizerTest)
endif()
