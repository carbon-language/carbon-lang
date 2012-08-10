# Build for the AddressSanitizer runtime support library.

set(ASAN_SOURCES
  asan_allocator.cc
  asan_globals.cc
  asan_interceptors.cc
  asan_linux.cc
  asan_mac.cc
  asan_malloc_linux.cc
  asan_malloc_mac.cc
  asan_malloc_win.cc
  asan_new_delete.cc
  asan_poisoning.cc
  asan_posix.cc
  asan_printf.cc
  asan_report.cc
  asan_rtl.cc
  asan_stack.cc
  asan_stats.cc
  asan_thread.cc
  asan_thread_registry.cc
  asan_win.cc
  )

include_directories(..)

set(ASAN_CFLAGS
  -fPIC
  -fno-exceptions
  -funwind-tables
  -fvisibility=hidden
  -fno-builtin
  -fomit-frame-pointer
  -O3
  )
if (SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  list(APPEND ASAN_CFLAGS -Wno-variadic-macros)
endif ()

if (APPLE)
  list(APPEND ASAN_CFLAGS -mmacosx-version-min=10.5)
endif()

set(ASAN_COMMON_DEFINITIONS
  ASAN_HAS_EXCEPTIONS=1
  ASAN_NEEDS_SEGV=1
  )

set(ASAN_RUNTIME_LIBRARIES)
if(APPLE)
  # Build universal binary on APPLE.
  add_library(clang_rt.asan_osx STATIC
    ${ASAN_SOURCES}
    $<TARGET_OBJECTS:RTInterception.osx>
    $<TARGET_OBJECTS:RTSanitizerCommon.osx>
    )
  set_target_compile_flags(clang_rt.asan_osx ${ASAN_CFLAGS})
  filter_available_targets(ASAN_TARGETS x86_64 i386)
  set_target_properties(clang_rt.asan_osx PROPERTIES
    OSX_ARCHITECTURES "${ASAN_TARGETS}")
  list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan_osx)
else()
  # Otherwise, build separate libraries for each target.
  if(CAN_TARGET_X86_64)
    add_library(clang_rt.asan-x86_64 STATIC
      ${ASAN_SOURCES}
      $<TARGET_OBJECTS:RTInterception.x86_64>
      $<TARGET_OBJECTS:RTSanitizerCommon.x86_64>
      )
    set_target_compile_flags(clang_rt.asan-x86_64
      ${ASAN_CFLAGS} ${TARGET_X86_64_CFLAGS}
      )
    list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan-x86_64)
  endif()
  if(CAN_TARGET_I386)
    add_library(clang_rt.asan-i386 STATIC
      ${ASAN_SOURCES}
      $<TARGET_OBJECTS:RTInterception.i386>
      $<TARGET_OBJECTS:RTSanitizerCommon.i386>
      )
    set_target_compile_flags(clang_rt.asan-i386
      ${ASAN_CFLAGS} ${TARGET_I386_CFLAGS}
      )
    list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan-i386)
  endif()
endif()

set_property(TARGET ${ASAN_RUNTIME_LIBRARIES} APPEND PROPERTY
  COMPILE_DEFINITIONS ${ASAN_COMMON_DEFINITIONS})
add_clang_runtime_static_library(${ASAN_RUNTIME_LIBRARIES})

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

# ASan output tests.
# FIXME: move all output tests from output_tests/ to lit_tests/ and get rid
# of the first directory.
add_subdirectory(lit_tests)
