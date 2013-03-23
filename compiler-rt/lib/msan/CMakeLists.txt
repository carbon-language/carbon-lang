include_directories(..)

# Runtime library sources and build flags.
set(MSAN_RTL_SOURCES
  msan.cc
  msan_allocator.cc
  msan_interceptors.cc
  msan_linux.cc
  msan_new_delete.cc
  msan_report.cc
  )
set(MSAN_RTL_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  -fno-rtti
  -fPIE
  # Prevent clang from generating libc calls.
  -ffreestanding)

# Static runtime library.
set(MSAN_RUNTIME_LIBRARIES)
set(arch "x86_64")
if(CAN_TARGET_${arch})
  add_compiler_rt_static_runtime(clang_rt.msan-${arch} ${arch}
    SOURCES ${MSAN_RTL_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
    CFLAGS ${MSAN_RTL_CFLAGS}
    SYMS msan.syms)
  list(APPEND MSAN_RUNTIME_LIBRARIES clang_rt.msan-${arch})
endif()

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(lit_tests)
