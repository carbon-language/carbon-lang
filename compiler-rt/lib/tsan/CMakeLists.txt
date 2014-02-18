# Build for the ThreadSanitizer runtime support library.

include_directories(..)

# SANITIZER_COMMON_CFLAGS contains -fPIC, but it's performance-critical for
# TSan runtime to be built with -fPIE to reduce the number of register spills.
set(TSAN_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  -fPIE
  -fno-rtti)

set(TSAN_RTL_CFLAGS ${TSAN_CFLAGS})
append_if(TSAN_RTL_CFLAGS COMPILER_RT_HAS_WFRAME_LARGER_THAN_FLAG -Wframe-larger-than=512)
append_if(TSAN_RTL_CFLAGS COMPILER_RT_HAS_WGLOBAL_CONSTRUCTORS_FLAG -Wglobal-constructors)
# FIXME: Add support for --sysroot=. compile flag:

if("${CMAKE_BUILD_TYPE}" EQUAL "Release")
  set(TSAN_COMMON_DEFINITIONS DEBUG=0)
else()
  set(TSAN_COMMON_DEFINITIONS DEBUG=1)
endif()

set(TSAN_SOURCES
  rtl/tsan_clock.cc
  rtl/tsan_flags.cc
  rtl/tsan_fd.cc
  rtl/tsan_ignoreset.cc
  rtl/tsan_interceptors.cc
  rtl/tsan_interface_ann.cc
  rtl/tsan_interface_atomic.cc
  rtl/tsan_interface.cc
  rtl/tsan_interface_java.cc
  rtl/tsan_md5.cc
  rtl/tsan_mman.cc
  rtl/tsan_mutex.cc
  rtl/tsan_mutexset.cc
  rtl/tsan_report.cc
  rtl/tsan_rtl.cc
  rtl/tsan_rtl_mutex.cc
  rtl/tsan_rtl_report.cc
  rtl/tsan_rtl_thread.cc
  rtl/tsan_stat.cc
  rtl/tsan_suppressions.cc
  rtl/tsan_symbolize.cc
  rtl/tsan_sync.cc)

if(APPLE)
  list(APPEND TSAN_SOURCES rtl/tsan_platform_mac.cc)
elseif(UNIX)
  # Assume Linux
  list(APPEND TSAN_SOURCES
    rtl/tsan_platform_linux.cc)
endif()

add_custom_target(tsan)
# TSan is currently supported on 64-bit Linux only.
if(CAN_TARGET_x86_64 AND UNIX AND NOT APPLE)
  set(TSAN_ASM_SOURCES rtl/tsan_rtl_amd64.S)
  # Pass ASM file directly to the C++ compiler.
  set_source_files_properties(${TSAN_ASM_SOURCES} PROPERTIES
    LANGUAGE C)
  set(arch "x86_64")
  add_compiler_rt_static_runtime(clang_rt.tsan-${arch} ${arch}
    SOURCES ${TSAN_SOURCES} ${TSAN_ASM_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    CFLAGS ${TSAN_RTL_CFLAGS}
    DEFS ${TSAN_COMMON_DEFINITIONS})
  add_sanitizer_rt_symbols(clang_rt.tsan-${arch} rtl/tsan.syms.extra)
  add_dependencies(tsan clang_rt.tsan-${arch}
    clang_rt.tsan-${arch}-symbols)
endif()

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
