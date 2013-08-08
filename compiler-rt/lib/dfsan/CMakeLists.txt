include_directories(..)

# Runtime library sources and build flags.
set(DFSAN_RTL_SOURCES
  dfsan.cc)

set(DFSAN_RTL_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  # Prevent clang from generating libc calls.
  -ffreestanding)

# Static runtime library.
set(DFSAN_RUNTIME_LIBRARIES)
set(arch "x86_64")
if(CAN_TARGET_${arch})
  add_compiler_rt_static_runtime(clang_rt.dfsan-${arch} ${arch}
    SOURCES ${DFSAN_RTL_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    CFLAGS ${DFSAN_RTL_CFLAGS} -fPIE)
  add_compiler_rt_static_runtime(clang_rt.dfsan-libc-${arch} ${arch}
    SOURCES ${DFSAN_RTL_SOURCES}
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            CFLAGS ${DFSAN_RTL_CFLAGS} -fPIC -DDFSAN_NOLIBC)
  list(APPEND DFSAN_RUNTIME_LIBRARIES clang_rt.dfsan-${arch})
endif()

add_subdirectory(lit_tests)
