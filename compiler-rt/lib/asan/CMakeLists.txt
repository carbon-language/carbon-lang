# Build for the AddressSanitizer runtime support library.

set(ASAN_SOURCES
  asan_allocator2.cc
  asan_activation.cc
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

include_directories(..)

if(ANDROID)
  include_directories(${COMPILER_RT_EXTRA_ANDROID_HEADERS})
endif()

if (NOT MSVC)
  set(ASAN_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    -fno-rtti)
else()
  set(ASAN_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    /GR-)
endif()

set(ASAN_COMMON_DEFINITIONS
  ASAN_HAS_EXCEPTIONS=1)

if(ANDROID)
  list(APPEND ASAN_COMMON_DEFINITIONS
    ASAN_LOW_MEMORY=1)
endif()

# Architectures supported by ASan.
filter_available_targets(ASAN_SUPPORTED_ARCH
  x86_64 i386 powerpc64)

# Compile ASan sources into an object library.
if(APPLE)
  foreach(os ${SANITIZER_COMMON_SUPPORTED_DARWIN_OS})
    add_compiler_rt_darwin_object_library(RTAsan ${os}
      ARCH ${ASAN_SUPPORTED_ARCH}
      SOURCES ${ASAN_SOURCES}
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
  endforeach()
elseif(ANDROID)
  add_library(RTAsan.arm.android OBJECT ${ASAN_SOURCES})
  set_target_compile_flags(RTAsan.arm.android ${ASAN_CFLAGS})
  set_property(TARGET RTAsan.arm.android APPEND PROPERTY
    COMPILE_DEFINITIONS ${ASAN_COMMON_DEFINITIONS})
else()
  foreach(arch ${ASAN_SUPPORTED_ARCH})
    add_compiler_rt_object_library(RTAsan ${arch}
      SOURCES ${ASAN_SOURCES} CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
  endforeach()
endif()

# Build ASan runtimes shipped with Clang.
set(ASAN_RUNTIME_LIBRARIES)
if(APPLE)
  foreach (os ${SANITIZER_COMMON_SUPPORTED_DARWIN_OS})
    # Dynamic lookup is needed because shadow scale and offset are
    # provided by the instrumented modules.
    set(ASAN_RUNTIME_LDFLAGS
        "-undefined dynamic_lookup")
    add_compiler_rt_darwin_dynamic_runtime(clang_rt.asan_${os}_dynamic ${os}
      ARCH ${ASAN_SUPPORTED_ARCH}
      SOURCES $<TARGET_OBJECTS:RTAsan.${os}>
              $<TARGET_OBJECTS:RTInterception.${os}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${os}>
              $<TARGET_OBJECTS:RTLSanCommon.${os}>
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS}
      LINKFLAGS ${ASAN_RUNTIME_LDFLAGS})
    list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan_${os}_dynamic)
  endforeach()

elseif(ANDROID)
  add_library(clang_rt.asan-arm-android SHARED
    $<TARGET_OBJECTS:RTAsan.arm.android>
    $<TARGET_OBJECTS:RTInterception.arm.android>
    $<TARGET_OBJECTS:RTSanitizerCommon.arm.android>)
  set_target_compile_flags(clang_rt.asan-arm-android
    ${ASAN_CFLAGS})
  set_property(TARGET clang_rt.asan-arm-android APPEND PROPERTY
    COMPILE_DEFINITIONS ${ASAN_COMMON_DEFINITIONS})
  target_link_libraries(clang_rt.asan-arm-android dl log)
  list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan-arm-android)
else()
  # Build separate libraries for each target.
  foreach(arch ${ASAN_SUPPORTED_ARCH})
    set(ASAN_RUNTIME_OBJECTS
      $<TARGET_OBJECTS:RTAsan.${arch}>
      $<TARGET_OBJECTS:RTInterception.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>)
    if (NOT WIN32)
      # We can't build Leak Sanitizer on Windows yet.
      list(APPEND ASAN_RUNTIME_OBJECTS $<TARGET_OBJECTS:RTLSanCommon.${arch}>)
    endif()

    add_compiler_rt_static_runtime(clang_rt.asan-${arch} ${arch}
      SOURCES ${ASAN_RUNTIME_OBJECTS}
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan-${arch})
    if (UNIX AND NOT ${arch} STREQUAL "i386")
      add_sanitizer_rt_symbols(clang_rt.asan-${arch} asan.syms.extra)
      list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan-${arch}-symbols)
    endif()

    if (WIN32)
      add_compiler_rt_static_runtime(clang_rt.asan_dll_thunk-${arch} ${arch}
      SOURCES asan_dll_thunk.cc
      CFLAGS ${ASAN_CFLAGS} -DASAN_DLL_THUNK
      DEFS ${ASAN_COMMON_DEFINITIONS})
      list(APPEND ASAN_RUNTIME_LIBRARIES clang_rt.asan_dll_thunk-${arch})
    endif()
  endforeach()
endif()

add_compiler_rt_resource_file(asan_blacklist asan_blacklist.txt)

# All ASan runtime dependencies.
add_custom_target(asan_runtime_libraries
  DEPENDS asan_blacklist ${ASAN_RUNTIME_LIBRARIES})

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(lit_tests)
