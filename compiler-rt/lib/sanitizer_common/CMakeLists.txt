# Build system for the common Sanitizer runtime support library components.
# These components are shared between AddressSanitizer and ThreadSanitizer.

set(SANITIZER_SOURCES
  sanitizer_allocator.cc
  sanitizer_common.cc
  sanitizer_coverage.cc
  sanitizer_flags.cc
  sanitizer_libc.cc
  sanitizer_libignore.cc
  sanitizer_linux.cc
  sanitizer_mac.cc
  sanitizer_platform_limits_linux.cc
  sanitizer_platform_limits_posix.cc
  sanitizer_posix.cc
  sanitizer_printf.cc
  sanitizer_procmaps_linux.cc
  sanitizer_procmaps_mac.cc
  sanitizer_stackdepot.cc
  sanitizer_stacktrace.cc
  sanitizer_suppressions.cc
  sanitizer_symbolizer.cc
  sanitizer_symbolizer_libbacktrace.cc
  sanitizer_symbolizer_win.cc
  sanitizer_tls_get_addr.cc
  sanitizer_thread_registry.cc
  sanitizer_win.cc)

set(SANITIZER_LIBCDEP_SOURCES
  sanitizer_common_libcdep.cc
  sanitizer_linux_libcdep.cc
  sanitizer_posix_libcdep.cc
  sanitizer_stacktrace_libcdep.cc
  sanitizer_stoptheworld_linux_libcdep.cc
  sanitizer_symbolizer_libcdep.cc
  sanitizer_symbolizer_posix_libcdep.cc)

# Explicitly list all sanitizer_common headers. Not all of these are
# included in sanitizer_common source files, but we need to depend on
# headers when building our custom unit tests.
set(SANITIZER_HEADERS
  sanitizer_allocator.h
  sanitizer_allocator_internal.h
  sanitizer_atomic.h
  sanitizer_atomic_clang.h
  sanitizer_atomic_msvc.h
  sanitizer_common.h
  sanitizer_common_interceptors.inc
  sanitizer_common_interceptors_ioctl.inc
  sanitizer_common_interceptors_format.inc
  sanitizer_common_syscalls.inc
  sanitizer_flags.h
  sanitizer_internal_defs.h
  sanitizer_lfstack.h
  sanitizer_libc.h
  sanitizer_libignore.h
  sanitizer_linux.h
  sanitizer_list.h
  sanitizer_mac.h
  sanitizer_mutex.h
  sanitizer_placement_new.h
  sanitizer_platform.h
  sanitizer_platform_interceptors.h
  sanitizer_platform_limits_posix.h
  sanitizer_procmaps.h
  sanitizer_quarantine.h
  sanitizer_report_decorator.h
  sanitizer_stackdepot.h
  sanitizer_stacktrace.h
  sanitizer_stoptheworld.h
  sanitizer_suppressions.h
  sanitizer_symbolizer.h
  sanitizer_symbolizer_libbacktrace.h
  sanitizer_syscall_generic.inc
  sanitizer_syscall_linux_x86_64.inc
  sanitizer_thread_registry.h)

set(SANITIZER_COMMON_DEFINITIONS)

if(ANDROID)
  list(APPEND SANITIZER_COMMON_DEFINITIONS
    SANITIZER_NEEDS_SEGV=0)
elseif(MSVC)
  list(APPEND SANITIZER_COMMON_DEFINITIONS
    SANITIZER_NEEDS_SEGV=0)
else()
  list(APPEND SANITIZER_COMMON_DEFINITIONS
    SANITIZER_NEEDS_SEGV=1)
endif()

if (NOT MSVC)
  set(SANITIZER_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    -fno-rtti)
else()
  set(SANITIZER_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    /GR-)
endif()

if(SUPPORTS_GLOBAL_CONSTRUCTORS_FLAG)
  list(APPEND SANITIZER_CFLAGS -Wglobal-constructors)
endif()

set(SANITIZER_RUNTIME_LIBRARIES)
if(APPLE)
  # Build universal binary on APPLE.
  foreach(os ${SANITIZER_COMMON_SUPPORTED_DARWIN_OS})
    add_compiler_rt_darwin_object_library(RTSanitizerCommon ${os}
      ARCH ${SANITIZER_COMMON_SUPPORTED_ARCH}
      SOURCES ${SANITIZER_SOURCES} ${SANITIZER_LIBCDEP_SOURCES}
      CFLAGS ${SANITIZER_CFLAGS}
      DEFS ${SANITIZER_COMMON_DEFINITIONS})
    list(APPEND SANITIZER_RUNTIME_LIBRARIES RTSanitizerCommon.${os})
  endforeach()
elseif(ANDROID)
  add_library(RTSanitizerCommon.arm.android OBJECT
    ${SANITIZER_SOURCES} ${SANITIZER_LIBCDEP_SOURCES})
  set_target_compile_flags(RTSanitizerCommon.arm.android
    ${SANITIZER_CFLAGS})
  set_property(TARGET RTSanitizerCommon.arm.android APPEND PROPERTY
    COMPILE_DEFINITIONS ${SANITIZER_COMMON_DEFINITIONS})
  list(APPEND SANITIZER_RUNTIME_LIBRARIES RTSanitizerCommon.arm.android)
else()
  # Otherwise, build separate libraries for each target.
  foreach(arch ${SANITIZER_COMMON_SUPPORTED_ARCH})
    add_compiler_rt_object_library(RTSanitizerCommon ${arch}
      SOURCES ${SANITIZER_SOURCES} CFLAGS ${SANITIZER_CFLAGS}
      DEFS ${SANITIZER_COMMON_DEFINITIONS})
    add_compiler_rt_object_library(RTSanitizerCommonLibc ${arch}
      SOURCES ${SANITIZER_LIBCDEP_SOURCES} CFLAGS ${SANITIZER_CFLAGS}
      DEFS ${SANITIZER_COMMON_DEFINITIONS})
    add_compiler_rt_static_runtime(clang_rt.san-${arch} ${arch}
      SOURCES $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
      CFLAGS ${SANITIZER_CFLAGS}
      DEFS ${SANITIZER_COMMON_DEFINITIONS})
    list(APPEND SANITIZER_RUNTIME_LIBRARIES RTSanitizerCommon.${arch}
                                            RTSanitizerCommonLibc.${arch})
  endforeach()
endif()

# Unit tests for common sanitizer runtime.
if(LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
