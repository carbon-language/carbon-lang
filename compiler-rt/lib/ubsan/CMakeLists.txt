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

set(UBSAN_RUNTIME_LIBRARIES)

if(CAN_TARGET_X86_64)
  add_library(clang_rt.ubsan-x86_64 STATIC
    ${UBSAN_SOURCES}
    $<TARGET_OBJECTS:RTSanitizerCommon.x86_64>
    )
  set_target_compile_flags(clang_rt.ubsan-x86_64
    ${UBSAN_CFLAGS} ${TARGET_X86_64_CFLAGS}
    )
  list(APPEND UBSAN_RUNTIME_LIBRARIES clang_rt.ubsan-x86_64)
endif()

if(CAN_TARGET_I386)
  add_library(clang_rt.ubsan-i386 STATIC
    ${UBSAN_SOURCES}
    $<TARGET_OBJECTS:RTSanitizerCommon.i386>
    )
  set_target_compile_flags(clang_rt.ubsan-i386
    ${UBSAN_CFLAGS} ${TARGET_I386_CFLAGS}
    )
  list(APPEND UBSAN_RUNTIME_LIBRARIES clang_rt.ubsan-i386)
endif()

set_property(TARGET ${UBSAN_RUNTIME_LIBRARIES} APPEND PROPERTY
  COMPILE_DEFINITIONS ${UBSAN_COMMON_DEFINITIONS})
add_clang_compiler_rt_libraries(${UBSAN_RUNTIME_LIBRARIES})

add_subdirectory(lit_tests)
