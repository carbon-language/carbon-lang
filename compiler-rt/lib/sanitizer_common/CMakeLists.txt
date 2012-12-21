# Build system for the common Sanitizer runtime support library components.
# These components are shared between AddressSanitizer and ThreadSanitizer.

set(SANITIZER_SOURCES
  sanitizer_allocator.cc
  sanitizer_common.cc
  sanitizer_flags.cc
  sanitizer_libc.cc
  sanitizer_linux.cc
  sanitizer_mac.cc
  sanitizer_posix.cc
  sanitizer_printf.cc
  sanitizer_stackdepot.cc
  sanitizer_stacktrace.cc
  sanitizer_symbolizer.cc
  sanitizer_symbolizer_itanium.cc
  sanitizer_symbolizer_linux.cc
  sanitizer_symbolizer_mac.cc
  sanitizer_symbolizer_win.cc
  sanitizer_win.cc
  )

# Explicitly list all sanitizer_common headers. Not all of these are
# included in sanitizer_common source files, but we need to depend on
# headers when building our custom unit tests.
set(SANITIZER_HEADERS
  sanitizer_allocator.h
  sanitizer_atomic_clang.h
  sanitizer_atomic_msvc.h
  sanitizer_atomic.h
  sanitizer_common.h
  sanitizer_common_interceptors.h
  sanitizer_flags.h
  sanitizer_internal_defs.h
  sanitizer_libc.h
  sanitizer_list.h
  sanitizer_mutex.h
  sanitizer_placement_new.h
  sanitizer_platform_interceptors.h
  sanitizer_procmaps.h
  sanitizer_report_decorator.h
  sanitizer_stackdepot.h
  sanitizer_stacktrace.h
  sanitizer_symbolizer.h
  )

set(SANITIZER_CFLAGS ${SANITIZER_COMMON_CFLAGS})

set(SANITIZER_RUNTIME_LIBRARIES)
if(APPLE)
  # Build universal binary on APPLE.
  add_library(RTSanitizerCommon.osx OBJECT ${SANITIZER_SOURCES})
  set_target_compile_flags(RTSanitizerCommon.osx ${SANITIZER_CFLAGS})
  filter_available_targets(SANITIZER_TARGETS x86_64 i386)
  set_target_properties(RTSanitizerCommon.osx PROPERTIES
    OSX_ARCHITECTURES "${SANITIZER_TARGETS}")
  list(APPEND SANITIZER_RUNTIME_LIBRARIES RTSanitizerCommon.osx)
else()
  # Otherwise, build separate libraries for each target.
  if(CAN_TARGET_X86_64)
    add_library(RTSanitizerCommon.x86_64 OBJECT ${SANITIZER_SOURCES})
    set_target_compile_flags(RTSanitizerCommon.x86_64
      ${SANITIZER_CFLAGS} ${TARGET_X86_64_CFLAGS})
    list(APPEND SANITIZER_RUNTIME_LIBRARIES RTSanitizerCommon.x86_64)
  endif()
  if(CAN_TARGET_I386)
    add_library(RTSanitizerCommon.i386 OBJECT ${SANITIZER_SOURCES})
    set_target_compile_flags(RTSanitizerCommon.i386
      ${SANITIZER_CFLAGS} ${TARGET_I386_CFLAGS})
    list(APPEND SANITIZER_RUNTIME_LIBRARIES RTSanitizerCommon.i386)
  endif()
  if(ANDROID)
    add_library(RTSanitizerCommon.arm.android OBJECT ${SANITIZER_SOURCES})
    set_target_compile_flags(RTSanitizerCommon.arm.android
      ${SANITIZER_CFLAGS})
    list(APPEND SANITIZER_RUNTIME_LIBRARIES RTSanitizerCommon.arm.android)
  endif()
endif()

# Unit tests for common sanitizer runtime.
if(LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
