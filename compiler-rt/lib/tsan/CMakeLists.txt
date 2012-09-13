# Build for the ThreadSanitizer runtime support library.

include_directories(..)

set(TSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
# FIXME: Add support for compile flags:
#   -Wframe-larger-than=512,
#   -Wglobal-constructors,
#   --sysroot=.

if("${CMAKE_BUILD_TYPE}" EQUAL "Release")
  set(TSAN_COMMON_DEFINITIONS DEBUG=0)
else()
  set(TSAN_COMMON_DEFINITIONS DEBUG=1)
endif()

add_subdirectory(rtl)

if(LLVM_INCLUDE_TESTS)
  add_custom_target(TsanUnitTests)
  set_target_properties(TsanUnitTests PROPERTIES
    FOLDER "TSan unittests")

  function(add_tsan_unittest testname)
    # Build unit tests only on 64-bit Linux.
    if(UNIX AND NOT APPLE
            AND CAN_TARGET_X86_64
            AND CMAKE_SIZEOF_VOID_P EQUAL 8
            AND NOT LLVM_BUILD_32_BITS)
      add_unittest(TsanUnitTests ${testname} ${ARGN})
      # Link with TSan runtime.
      target_link_libraries(${testname} clang_rt.tsan-x86_64)
      # Build tests with PIE and debug info.
      set_property(TARGET ${testname} APPEND_STRING
        PROPERTY COMPILE_FLAGS " -fPIE -g")
      set_property(TARGET ${testname} APPEND_STRING
        PROPERTY LINK_FLAGS " -pie")
    endif()
  endfunction()

  include_directories(rtl)
  # There are two groups of unit tests: rtl_tests and unit_tests.
  add_subdirectory(rtl_tests)
  add_subdirectory(unit_tests)
endif()

# FIXME: Support TSan runtime tests, unit tests and output tests.
