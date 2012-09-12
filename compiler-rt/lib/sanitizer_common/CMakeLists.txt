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
  sanitizer_symbolizer_linux.cc
  sanitizer_symbolizer_mac.cc
  sanitizer_symbolizer_win.cc
  sanitizer_win.cc
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
  # Build stand-alone static sanitizer runtime for use in unit tests.
  add_library(RTSanitizerCommon.test STATIC ${SANITIZER_SOURCES})
  # Build test library with debug info.
  set_target_compile_flags(RTSanitizerCommon.test
    ${SANITIZER_CFLAGS} -g)
  set_target_properties(RTSanitizerCommon.test PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_subdirectory(tests)
endif()
