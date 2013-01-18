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
  add_library(clang_rt.ubsan_osx STATIC
    ${UBSAN_SOURCES}
    $<TARGET_OBJECTS:RTSanitizerCommon.osx>
    )
  set_target_compile_flags(clang_rt.ubsan_osx ${UBSAN_CFLAGS})
  set_target_properties(clang_rt.ubsan_osx PROPERTIES
    OSX_ARCHITECTURES "${UBSAN_SUPPORTED_ARCH}")
  list(APPEND UBSAN_RUNTIME_LIBRARIES clang_rt.ubsan_osx)
else()
  # Build separate libraries for each target.
  foreach(arch ${UBSAN_SUPPORTED_ARCH})
    add_library(clang_rt.ubsan-${arch} STATIC
      ${UBSAN_SOURCES}
      $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      )
    set_target_compile_flags(clang_rt.ubsan-${arch}
      ${UBSAN_CFLAGS} ${TARGET_${arch}_CFLAGS}
      )
    list(APPEND UBSAN_RUNTIME_LIBRARIES clang_rt.ubsan-${arch})
  endforeach()
endif()


set_property(TARGET ${UBSAN_RUNTIME_LIBRARIES} APPEND PROPERTY
  COMPILE_DEFINITIONS ${UBSAN_COMMON_DEFINITIONS})
add_clang_compiler_rt_libraries(${UBSAN_RUNTIME_LIBRARIES})

add_subdirectory(lit_tests)
