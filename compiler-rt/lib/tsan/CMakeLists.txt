# Build for the ThreadSanitizer runtime support library.

include_directories(..)

set(TSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
# SANITIZER_COMMON_CFLAGS contains -fPIC, but it's performance-critical for
# TSan runtime to be built with -fPIE to reduce the number of register spills.
append_list_if(COMPILER_RT_HAS_FPIE_FLAG -fPIE TSAN_CFLAGS)
append_no_rtti_flag(TSAN_CFLAGS)

set(TSAN_RTL_CFLAGS ${TSAN_CFLAGS})
append_list_if(COMPILER_RT_HAS_WFRAME_LARGER_THAN_FLAG -Wframe-larger-than=512 TSAN_RTL_CFLAGS)
append_list_if(COMPILER_RT_HAS_WGLOBAL_CONSTRUCTORS_FLAG -Wglobal-constructors TSAN_RTL_CFLAGS)
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
  rtl/tsan_stack_trace.cc
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

set(TSAN_HEADERS
  rtl/tsan_clock.h
  rtl/tsan_defs.h
  rtl/tsan_dense_alloc.h
  rtl/tsan_fd.h
  rtl/tsan_flags.h
  rtl/tsan_ignoreset.h
  rtl/tsan_interface_ann.h
  rtl/tsan_interface.h
  rtl/tsan_interface_inl.h
  rtl/tsan_interface_java.h
  rtl/tsan_mman.h
  rtl/tsan_mutex.h
  rtl/tsan_mutexset.h
  rtl/tsan_platform.h
  rtl/tsan_report.h
  rtl/tsan_rtl.h
  rtl/tsan_stack_trace.h
  rtl/tsan_stat.h
  rtl/tsan_suppressions.h
  rtl/tsan_symbolize.h
  rtl/tsan_sync.h
  rtl/tsan_trace.h
  rtl/tsan_update_shadow_word_inl.h
  rtl/tsan_vector.h)

set(TSAN_RUNTIME_LIBRARIES)
add_custom_target(tsan)
# TSan is currently supported on 64-bit Linux only.
if(CAN_TARGET_x86_64 AND UNIX AND NOT APPLE)
  set(TSAN_ASM_SOURCES rtl/tsan_rtl_amd64.S)
  # Pass ASM file directly to the C++ compiler.
  set_source_files_properties(${TSAN_ASM_SOURCES} PROPERTIES
    LANGUAGE C)
  set(arch "x86_64")
  add_compiler_rt_runtime(clang_rt.tsan-${arch} ${arch} STATIC
    SOURCES ${TSAN_SOURCES} ${TSAN_ASM_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    CFLAGS ${TSAN_RTL_CFLAGS}
    DEFS ${TSAN_COMMON_DEFINITIONS})
  list(APPEND TSAN_RUNTIME_LIBRARIES clang_rt.tsan-${arch})
  add_sanitizer_rt_symbols(clang_rt.tsan-${arch} rtl/tsan.syms.extra)
  add_dependencies(tsan clang_rt.tsan-${arch}
    clang_rt.tsan-${arch}-symbols)
endif()

add_dependencies(compiler-rt tsan)

# Build libcxx instrumented with TSan.
if(TSAN_SUPPORTED_ARCH AND
   COMPILER_RT_HAS_LIBCXX_SOURCES AND
   COMPILER_RT_TEST_COMPILER_ID STREQUAL "Clang")
  set(LIBCXX_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libcxx_tsan)
  add_custom_libcxx(libcxx_tsan ${LIBCXX_PREFIX}
    DEPS ${TSAN_RUNTIME_LIBRARIES}
    CFLAGS -fsanitize=thread)
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
