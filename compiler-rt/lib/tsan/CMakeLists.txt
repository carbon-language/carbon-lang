# Build for the ThreadSanitizer runtime support library.

include_directories(..)

# SANITIZER_COMMON_CFLAGS contains -fPIC, but it's performance-critical for
# TSan runtime to be built with -fPIE to reduce the number of register spills.
set(TSAN_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  -fPIE
  -fno-rtti)
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
  add_subdirectory(tests)
endif()
add_subdirectory(lit_tests)
