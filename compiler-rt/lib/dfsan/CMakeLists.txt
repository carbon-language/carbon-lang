include_directories(..)

# Runtime library sources and build flags.
set(DFSAN_RTL_SOURCES
  dfsan.cc
  dfsan_custom.cc
  dfsan_interceptors.cc)
set(DFSAN_COMMON_CFLAGS ${SANITIZER_COMMON_CFLAGS})
# Prevent clang from generating libc calls.
append_if(DFSAN_COMMON_CFLAGS COMPILER_RT_HAS_FFREESTANDING_FLAG -ffreestanding)

# Static runtime library.
add_custom_target(dfsan)
set(arch "x86_64")
if(CAN_TARGET_${arch})
  set(DFSAN_CFLAGS ${DFSAN_COMMON_CFLAGS})
  append_if(DFSAN_CFLAGS COMPILER_RT_HAS_FPIE_FLAG -fPIE)
  add_compiler_rt_static_runtime(clang_rt.dfsan-${arch} ${arch}
    SOURCES ${DFSAN_RTL_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    CFLAGS ${DFSAN_CFLAGS})
  set(DFSAN_NOLIBC_CFLAGS ${DFSAN_COMMON_CFLAGS} -DDFSAN_NOLIBC)
  add_compiler_rt_static_runtime(clang_rt.dfsan-libc-${arch} ${arch}
    SOURCES ${DFSAN_RTL_SOURCES}
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            CFLAGS ${DFSAN_NOLIBC_CFLAGS})
  add_sanitizer_rt_symbols(clang_rt.dfsan-${arch} dfsan.syms.extra)
  add_dependencies(dfsan
    clang_rt.dfsan-${arch}
    clang_rt.dfsan-${arch}-symbols)
endif()

add_custom_target(dfsan_abilist ALL
                                SOURCES ${CLANG_RESOURCE_DIR}/dfsan_abilist.txt)
add_custom_command(OUTPUT ${CLANG_RESOURCE_DIR}/dfsan_abilist.txt
                   VERBATIM
                   COMMAND
                     cat ${CMAKE_CURRENT_SOURCE_DIR}/done_abilist.txt
                         ${CMAKE_CURRENT_SOURCE_DIR}/libc_ubuntu1204_abilist.txt
                         > ${CLANG_RESOURCE_DIR}/dfsan_abilist.txt
                   DEPENDS done_abilist.txt libc_ubuntu1204_abilist.txt)
add_dependencies(dfsan dfsan_abilist)
install(FILES ${CLANG_RESOURCE_DIR}/dfsan_abilist.txt
        DESTINATION ${LIBCLANG_INSTALL_PATH})
