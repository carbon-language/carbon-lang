include(CompilerRTCompile)
include(CompilerRTUnittests)

set(SANITIZER_UNITTESTS
  sanitizer_allocator_test.cc
  sanitizer_common_test.cc
  sanitizer_flags_test.cc
  sanitizer_libc_test.cc
  sanitizer_list_test.cc
  sanitizer_printf_test.cc
  sanitizer_stackdepot_test.cc
  sanitizer_test_main.cc
  )

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
  set(SANITIZER_TEST_CFLAGS ${COMPILER_RT_GTEST_INCLUDE_CFLAGS}
                            -I${COMPILER_RT_SOURCE_DIR}/include
                            -I${COMPILER_RT_SOURCE_DIR}/lib
                            -I${COMPILER_RT_SOURCE_DIR}/lib/sanitizer_common
                            -O2 -g ${TARGET_FLAGS})
  set(SANITIZER_TEST_LINK_FLAGS -lstdc++ -lpthread ${TARGET_FLAGS})
  set(SANITIZER_TEST_OBJECTS)
  foreach(source ${SANITIZER_TEST_SOURCES})
    get_filename_component(basename ${source} NAME)
    set(output_obj "${basename}.${arch}.o")
    clang_compile(${output_obj} ${source}
                  CFLAGS ${SANITIZER_TEST_CFLAGS}
                  DEPS gtest ${SANITIZER_RUNTIME_LIBRARIES})
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
                       LINK_FLAGS ${SANITIZER_TEST_LINK_FLAGS})
endmacro()

if("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND UNIX AND NOT ANDROID)
  if(APPLE)
    add_sanitizer_common_lib("RTSanitizerCommon.test.osx"
                             $<TARGET_OBJECTS:RTSanitizerCommon.osx>)
  else()
    if(CAN_TARGET_X86_64)
      add_sanitizer_common_lib("RTSanitizerCommon.test.x86_64"
                               $<TARGET_OBJECTS:RTSanitizerCommon.x86_64>)
    endif()
    if(CAN_TARGET_I386)
      add_sanitizer_common_lib("RTSanitizerCommon.test.i386"
                               $<TARGET_OBJECTS:RTSanitizerCommon.i386>)
    endif()
  endif()
  if(CAN_TARGET_X86_64)
    add_sanitizer_tests_for_arch(x86_64)
  endif()
  if(CAN_TARGET_I386)
    add_sanitizer_tests_for_arch(i386)
  endif()
endif()

# Run unittests as a part of lit testsuite.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_lit_testsuite(check-sanitizer "Running sanitizer library unittests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS SanitizerUnitTests
  )
set_target_properties(check-sanitizer PROPERTIES FOLDER "Sanitizer unittests")
