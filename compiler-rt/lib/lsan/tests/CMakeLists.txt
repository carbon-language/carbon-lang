include(CheckCXXCompilerFlag)
include(CompilerRTCompile)
include(CompilerRTLink)

include_directories(..)
include_directories(../..)

set(LSAN_TESTS_SRC
  lsan_test.cc)

set(LSAN_TESTS_CFLAGS
  ${LSAN_CFLAGS}
  ${COMPILER_RT_GTEST_INCLUDE_CFLAGS}
  -I${COMPILER_RT_SOURCE_DIR}/lib)

add_custom_target(LsanTests)
set_target_properties(LsanTests PROPERTIES
  FOLDER "LSan unittests")

set(LSAN_LOADABLE_SRC "lsan_tls_loadable.cc")

# Compile source for the given architecture, using compiler
# options in ${ARGN}, and add it to the object list.
macro(lsan_compile obj_list source arch)
  get_filename_component(basename ${source} NAME)
  set(output_obj "${basename}.${arch}.o")
  get_target_flags_for_arch(${arch} TARGET_CFLAGS)
  clang_compile(${output_obj} ${source}
                CFLAGS ${ARGN} ${TARGET_CFLAGS}
                DEPS gtest)
  list(APPEND ${obj_list} ${output_obj})
endmacro()

function(add_lsan_test testname arch)
  set(testname_arch ${testname}-${arch}-Test)
  get_target_flags_for_arch(${arch} TARGET_LINKFLAGS)
  add_unittest(LsanTests ${testname_arch} ${ARGN})
  target_link_libraries(${testname_arch} "clang_rt.lsan-${arch}")
  set_target_compile_flags(${testname_arch} ${LSAN_TESTS_CFLAGS})
  set_target_link_flags(${testname_arch} ${TARGET_LINKFLAGS})
endfunction()

macro(add_lsan_tests_for_arch arch)
  set(LSAN_TESTS_OBJ)
  lsan_compile(LSAN_TESTS_OBJ ${LSAN_TESTS_SRC} ${arch} ${LSAN_TESTS_CFLAGS}
      -I${LSAN_SRC_DIR})
  add_lsan_test(Lsan ${arch} ${LSAN_TESTS_OBJ})
  add_library("lsan_tls_loadable-${arch}" SHARED ${LSAN_LOADABLE_SRC})
  set_property(TARGET "lsan_tls_loadable-${arch}" PROPERTY
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(Lsan-${arch}-Test  lsan_tls_loadable-${arch})
endmacro()

# Build tests for 64-bit Linux only.
if(UNIX AND NOT APPLE AND CAN_TARGET_x86_64)
  add_lsan_tests_for_arch(x86_64)
endif()

