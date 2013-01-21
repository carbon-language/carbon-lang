# Build for the undefined behavior sanitizer runtime support library.

set(UBSAN_SOURCES
  ubsan_diag.cc
  ubsan_handlers.cc
  ubsan_handlers_cxx.cc
  ubsan_type_hash.cc
  ubsan_value.cc
  )

include_directories(..)

set(UBSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})

filter_available_targets(UBSAN_SUPPORTED_ARCH
  x86_64 i386)

set(UBSAN_RUNTIME_LIBRARIES)

if(APPLE)
  # Build universal binary on APPLE.
  add_compiler_rt_osx_static_runtime(clang_rt.ubsan_osx
    ARCH ${UBSAN_SUPPORTED_ARCH}
    SOURCES ${UBSAN_SOURCES}
            $<TARGET_OBJECTS:RTSanitizerCommon.osx>
    CFLAGS ${UBSAN_CFLAGS})
  list(APPEND UBSAN_RUNTIME_LIBRARIES clang_rt.ubsan_osx)
else()
  # Build separate libraries for each target.
  foreach(arch ${UBSAN_SUPPORTED_ARCH})
    add_compiler_rt_static_runtime(clang_rt.ubsan-${arch} ${arch}
      SOURCES ${UBSAN_SOURCES}
              $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      CFLAGS ${UBSAN_CFLAGS})
    list(APPEND UBSAN_RUNTIME_LIBRARIES clang_rt.ubsan-${arch})
  endforeach()
endif()

add_subdirectory(lit_tests)
