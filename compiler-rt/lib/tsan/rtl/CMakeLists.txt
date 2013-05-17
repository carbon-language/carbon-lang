set(TSAN_SOURCES
  tsan_clock.cc
  tsan_flags.cc
  tsan_fd.cc
  tsan_interceptors.cc
  tsan_interface_ann.cc
  tsan_interface_atomic.cc
  tsan_interface.cc
  tsan_interface_java.cc
  tsan_md5.cc
  tsan_mman.cc
  tsan_mutex.cc
  tsan_mutexset.cc
  tsan_report.cc
  tsan_rtl.cc
  tsan_rtl_mutex.cc
  tsan_rtl_report.cc
  tsan_rtl_thread.cc
  tsan_stat.cc
  tsan_suppressions.cc
  tsan_symbolize.cc
  tsan_sync.cc
  )

if(APPLE)
  list(APPEND TSAN_SOURCES tsan_platform_mac.cc)
elseif(UNIX)
  # Assume Linux
  list(APPEND TSAN_SOURCES
    tsan_platform_linux.cc
    tsan_symbolize_addr2line_linux.cc)
endif()

set(TSAN_RUNTIME_LIBRARIES)
# TSan is currently supported on 64-bit Linux only.
if(CAN_TARGET_x86_64 AND UNIX AND NOT APPLE)
  set(TSAN_ASM_SOURCES tsan_rtl_amd64.S)
  # Pass ASM file directly to the C++ compiler.
  set_source_files_properties(${TSAN_ASM_SOURCES} PROPERTIES
    LANGUAGE C)
  set(arch "x86_64")
  add_compiler_rt_static_runtime(clang_rt.tsan-${arch} ${arch}
    SOURCES ${TSAN_SOURCES} ${TSAN_ASM_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    CFLAGS ${TSAN_CFLAGS}
    DEFS ${TSAN_COMMON_DEFINITIONS}
    SYMS tsan.syms)
  list(APPEND TSAN_RUNTIME_LIBRARIES clang_rt.tsan-${arch})
endif()
