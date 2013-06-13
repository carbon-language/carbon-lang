include(CheckCXXCompilerFlag)
include(CompilerRTCompile)
include(CompilerRTLink)

include_directories(..)
include_directories(../..)

set(LSAN_TESTS_SRC
  lsan_dummy_unittest.cc)

set(LSAN_TESTS_CFLAGS
  ${LSAN_CFLAGS}
  ${COMPILER_RT_GTEST_INCLUDE_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/lib
  -I${LSAN_SRC_DIR})

set(LSAN_TEST_LINK_FLAGS_COMMON
  -lstdc++ -ldl -lpthread -lm)

add_custom_target(LsanUnitTests)
set_target_properties(LsanUnitTests PROPERTIES
  FOLDER "LSan unit tests")

# Compile source for the given architecture, using compiler
# options in ${ARGN}, and add it to the object list.
macro(lsan_compile obj_list source arch)
  get_filename_component(basename ${source} NAME)
  set(output_obj "${basename}.${arch}.o")
  get_target_flags_for_arch(${arch} TARGET_CFLAGS)
  clang_compile(${output_obj} ${source}
                CFLAGS ${ARGN} ${TARGET_CFLAGS}
                DEPS gtest ${LSAN_RUNTIME_LIBRARIES})
  list(APPEND ${obj_list} ${output_obj})
endmacro()

function(add_lsan_test test_suite test_name arch)
  get_target_flags_for_arch(${arch} TARGET_LINK_FLAGS)
  add_compiler_rt_test(${test_suite} ${test_name}
                       OBJECTS ${ARGN}
                       DEPS ${LSAN_RUNTIME_LIBRARIES} ${ARGN}
                       LINK_FLAGS ${LSAN_TEST_LINK_FLAGS_COMMON}
                                  ${TARGET_LINK_FLAGS})
endfunction()

macro(add_lsan_tests_for_arch arch)
  set(LSAN_TESTS_OBJ)
  set(LSAN_TEST_SOURCES ${LSAN_TESTS_SRC}
                        ${COMPILER_RT_GTEST_SOURCE})
  foreach(source ${LSAN_TEST_SOURCES})
    lsan_compile(LSAN_TESTS_OBJ ${source} ${arch} ${LSAN_TESTS_CFLAGS})
  endforeach()
  add_lsan_test(LsanUnitTests Lsan-${arch}-Test ${arch} ${LSAN_TESTS_OBJ})
endmacro()

# Build tests for 64-bit Linux only.
if(UNIX AND NOT APPLE AND CAN_TARGET_x86_64)
  add_lsan_tests_for_arch(x86_64)
endif()
