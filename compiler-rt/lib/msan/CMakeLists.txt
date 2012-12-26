include_directories(..)

# Runtime library sources and build flags.
set(MSAN_RTL_SOURCES
  msan.cc
  msan_allocator.cc
  msan_interceptors.cc
  msan_linux.cc
  msan_new_delete.cc
  msan_platform_limits_posix.cc
  msan_report.cc
  )
set(MSAN_RTL_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  -fPIE
  # Prevent clang from generating libc calls.
  -ffreestanding)

# Static runtime library.
set(MSAN_RUNTIME_LIBRARIES)
add_library(clang_rt.msan-x86_64 STATIC
  ${MSAN_RTL_SOURCES}
  $<TARGET_OBJECTS:RTInterception.x86_64>
  $<TARGET_OBJECTS:RTSanitizerCommon.x86_64>
  )
set_target_compile_flags(clang_rt.msan-x86_64
  ${MSAN_RTL_CFLAGS} ${TARGET_X86_64_CFLAGS}
  )
list(APPEND MSAN_RUNTIME_LIBRARIES clang_rt.msan-x86_64)

add_clang_compiler_rt_libraries(${MSAN_RUNTIME_LIBRARIES})

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
