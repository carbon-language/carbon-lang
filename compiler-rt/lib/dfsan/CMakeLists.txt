include_directories(..)

# Runtime library sources and build flags.
set(DFSAN_RTL_SOURCES
  dfsan.cc
  dfsan_custom.cc
  dfsan_interceptors.cc)
set(DFSAN_RTL_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  # Prevent clang from generating libc calls.
  -ffreestanding)

# Static runtime library.
add_custom_target(dfsan)
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
