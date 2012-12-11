# Build for the MemorySanitizer runtime support library.

set(MSAN_SOURCES
  msan.cc
  msan_allocator.cc
  msan_interceptors.cc
  msan_linux.cc
  msan_new_delete.cc
  msan_platform_limits_posix.cc
  )

include_directories(..)

set(MSAN_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  -fPIE
  -ffreestanding
  -g
  -fno-omit-frame-pointer)
set(MSAN_COMMON_DEFINITIONS)

set(MSAN_RUNTIME_LIBRARIES)
add_library(clang_rt.msan-x86_64 STATIC
  ${MSAN_SOURCES}
  $<TARGET_OBJECTS:RTInterception.x86_64>
  $<TARGET_OBJECTS:RTSanitizerCommon.x86_64>
  )
set_target_compile_flags(clang_rt.msan-x86_64
  ${MSAN_CFLAGS} ${TARGET_X86_64_CFLAGS}
  )
list(APPEND MSAN_RUNTIME_LIBRARIES clang_rt.msan-x86_64)

set_property(TARGET ${MSAN_RUNTIME_LIBRARIES} APPEND PROPERTY
  COMPILE_DEFINITIONS ${MSAN_COMMON_DEFINITIONS})
add_clang_compiler_rt_libraries(${MSAN_RUNTIME_LIBRARIES})
