# Build for the AddressSanitizer runtime support library.

set(ASAN_SOURCES
  asan_allocator2.cc
  asan_fake_stack.cc
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
  asan_preinit.cc
  asan_report.cc
  asan_rtl.cc
  asan_stack.cc
  asan_stats.cc
  asan_thread.cc
  asan_win.cc)

set(ASAN_DYLIB_SOURCES
  ${ASAN_SOURCES})

include_directories(..)

set(ASAN_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  -fno-rtti)

set(ASAN_COMMON_DEFINITIONS
  ASAN_HAS_EXCEPTIONS=1)

if(ANDROID)
  list(APPEND ASAN_COMMON_DEFINITIONS
    ASAN_FLEXIBLE_MAPPING_AND_OFFSET=0
    ASAN_NEEDS_SEGV=0
    ASAN_LOW_MEMORY=1)
else()
  list(APPEND ASAN_COMMON_DEFINITIONS
    ASAN_FLEXIBLE_MAPPING_AND_OFFSET=1
    ASAN_NEEDS_SEGV=1)
endif()

# Architectures supported by ASan.
filter_available_targets(ASAN_SUPPORTED_ARCH
  x86_64 i386 powerpc64)

set(ASAN_RUNTIME_LIBRARIES)
if(APPLE)
  # Build universal binary on APPLE.
  add_compiler_rt_osx_dynamic_runtime(clang_rt.asan_osx_dynamic
    ARCH ${ASAN_SUPPORTED_ARCH}
    SOURCES ${ASAN_DYLIB_SOURCES}
            $<TARGET_OBJECTS:RTInterception.osx>
            $<TARGET_OBJECTS:RTSanitizerCommon.osx>
            $<TARGET_OBJECTS:RTLSanCommon.osx>
    CFLAGS ${ASAN_CFLAGS}
    DEFS ${ASAN_COMMON_DEFINITIONS}
    # Dynamic lookup is needed because shadow scale and offset are
    # provided by the instrumented modules.
    LINKFLAGS "-framework Foundation"
              "-undefined dynamic_lookup")
  list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan_osx_dynamic)
elseif(ANDROID)
  add_library(clang_rt.asan-arm-android SHARED
    ${ASAN_SOURCES}
    $<TARGET_OBJECTS:RTInterception.arm.android>
    $<TARGET_OBJECTS:RTSanitizerCommon.arm.android>
    )
  set_target_compile_flags(clang_rt.asan-arm-android
    ${ASAN_CFLAGS})
  set_property(TARGET clang_rt.asan-arm-android APPEND PROPERTY
    COMPILE_DEFINITIONS ${ASAN_COMMON_DEFINITIONS})
  target_link_libraries(clang_rt.asan-arm-android dl)
  list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan-arm-android)
else()
  # Otherwise, build separate libraries for each target.
  foreach(arch ${ASAN_SUPPORTED_ARCH})
    add_compiler_rt_static_runtime(clang_rt.asan-${arch} ${arch}
      SOURCES ${ASAN_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
              $<TARGET_OBJECTS:RTLSanCommon.${arch}>
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS}
      SYMS asan.syms)
    list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan-${arch})
  endforeach()
endif()

add_compiler_rt_resource_file(asan_blacklist asan_blacklist.txt)

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(lit_tests)
