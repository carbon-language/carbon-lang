include_directories(..)

set(LSAN_CFLAGS
  ${SANITIZER_COMMON_CFLAGS})

set(LSAN_SOURCES
  lsan_interceptors.cc
  lsan_allocator.cc
  lsan_thread.cc
  lsan.cc
  lsan_common.cc
  lsan_common_linux.cc)

set(LSAN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(LSAN_RUNTIME_LIBRARIES)
set(arch "x86_64")
add_compiler_rt_static_runtime(clang_rt.lsan-${arch} ${arch}
  SOURCES ${LSAN_SOURCES}
          $<TARGET_OBJECTS:RTInterception.${arch}>
          $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
          $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
  CFLAGS ${LSAN_CFLAGS})
list(APPEND LSAN_RUNTIME_LIBRARIES clang_rt.lsan-${arch})

add_subdirectory(tests)
