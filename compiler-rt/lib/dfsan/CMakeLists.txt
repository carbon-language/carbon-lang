include_directories(..)

# Runtime library sources and build flags.
set(DFSAN_RTL_SOURCES
  dfsan.cc
  dfsan_custom.cc
  dfsan_interceptors.cc)
set(DFSAN_COMMON_CFLAGS ${SANITIZER_COMMON_CFLAGS})
# Prevent clang from generating libc calls.
append_if(COMPILER_RT_HAS_FFREESTANDING_FLAG -ffreestanding DFSAN_COMMON_CFLAGS)

# Static runtime library.
add_custom_target(dfsan)
set(arch "x86_64")
if(CAN_TARGET_${arch})
  set(DFSAN_CFLAGS ${DFSAN_COMMON_CFLAGS})
  append_if(COMPILER_RT_HAS_FPIE_FLAG -fPIE DFSAN_CFLAGS)
  add_compiler_rt_runtime(clang_rt.dfsan-${arch} ${arch} STATIC
    SOURCES ${DFSAN_RTL_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    CFLAGS ${DFSAN_CFLAGS})
  set(DFSAN_NOLIBC_CFLAGS ${DFSAN_COMMON_CFLAGS} -DDFSAN_NOLIBC)
  add_compiler_rt_runtime(clang_rt.dfsan-libc-${arch} ${arch} STATIC
    SOURCES ${DFSAN_RTL_SOURCES}
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            CFLAGS ${DFSAN_NOLIBC_CFLAGS})
  add_sanitizer_rt_symbols(clang_rt.dfsan-${arch} dfsan.syms.extra)
  add_dependencies(dfsan
    clang_rt.dfsan-${arch}
    clang_rt.dfsan-${arch}-symbols)
endif()

set(dfsan_abilist_filename ${COMPILER_RT_OUTPUT_DIR}/dfsan_abilist.txt)
add_custom_target(dfsan_abilist ALL
  DEPENDS ${dfsan_abilist_filename})
add_custom_command(OUTPUT ${dfsan_abilist_filename}
                   VERBATIM
                   COMMAND
                     cat ${CMAKE_CURRENT_SOURCE_DIR}/done_abilist.txt
                         ${CMAKE_CURRENT_SOURCE_DIR}/libc_ubuntu1204_abilist.txt
                         > ${dfsan_abilist_filename}
                   DEPENDS done_abilist.txt libc_ubuntu1204_abilist.txt)
add_dependencies(dfsan dfsan_abilist)
install(FILES ${dfsan_abilist_filename}
        DESTINATION ${COMPILER_RT_INSTALL_PATH})

add_dependencies(compiler-rt dfsan)
