set(TSAN_SOURCES
  tsan_clock.cc
  tsan_flags.cc
  tsan_fd.cc
  tsan_interceptors.cc
  tsan_interface_ann.cc
  tsan_interface_atomic.cc
  tsan_interface.cc
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
if(CAN_TARGET_X86_64 AND UNIX AND NOT APPLE)
  set(TSAN_ASM_SOURCES tsan_rtl_amd64.S)
  # Pass ASM file directly to the C++ compiler.
  set_source_files_properties(${TSAN_ASM_SOURCES} PROPERTIES
    LANGUAGE C
    )
  add_library(clang_rt.tsan-x86_64 STATIC
    ${TSAN_SOURCES}
    ${TSAN_ASM_SOURCES}
    $<TARGET_OBJECTS:RTInterception.x86_64>
    $<TARGET_OBJECTS:RTSanitizerCommon.x86_64>
    )
  set_target_compile_flags(clang_rt.tsan-x86_64
    ${TSAN_CFLAGS} ${TARGET_X86_64_CFLAGS}
    )
  list(APPEND TSAN_RUNTIME_LIBRARIES clang_rt.tsan-x86_64)
endif()

if(TSAN_RUNTIME_LIBRARIES)
  set_property(TARGET ${TSAN_RUNTIME_LIBRARIES} APPEND PROPERTY
    COMPILE_DEFINITIONS ${TSAN_COMMON_DEFINITIONS})
  add_clang_compiler_rt_libraries(${TSAN_RUNTIME_LIBRARIES})
endif()
