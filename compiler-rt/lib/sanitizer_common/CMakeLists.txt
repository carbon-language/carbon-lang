# Build system for the common Sanitizer runtime support library components.
# These components are shared between AddressSanitizer and ThreadSanitizer.

set(SANITIZER_SOURCES
  sanitizer_allocator.cc
  sanitizer_common.cc
  sanitizer_libc.cc
  sanitizer_linux.cc
  sanitizer_mac.cc
  sanitizer_posix.cc
  sanitizer_printf.cc
  sanitizer_symbolizer.cc
  sanitizer_win.cc
  )

if(CAN_TARGET_X86_64)
  add_library(clang_rt.sanitizer-x86_64 STATIC ${SANITIZER_SOURCES})
  set_target_properties(clang_rt.sanitizer-x86_64 PROPERTIES COMPILE_FLAGS "${TARGET_X86_64_CFLAGS}")
endif()
if(CAN_TARGET_I386)
  add_library(clang_rt.sanitizer-i386 STATIC ${SANITIZER_SOURCES})
  set_target_properties(clang_rt.sanitizer-i386 PROPERTIES COMPILE_FLAGS "${TARGET_I386_CFLAGS}")
endif()
