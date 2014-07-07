# Build for the AddressSanitizer runtime support library.

if(APPLE)
# Don't set rpath for the ASan libraries. Developers are encouraged to ship
# their binaries together with the corresponding ASan runtime libraries,
# so they'll anyway need to fix the rpath and the install name.
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
endif()

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
  asan_poisoning.cc
  asan_posix.cc
  asan_report.cc
  asan_rtl.cc
  asan_stack.cc
  asan_stats.cc
  asan_thread.cc
  asan_win.cc)

set(ASAN_CXX_SOURCES
  asan_new_delete.cc)

set(ASAN_PREINIT_SOURCES
  asan_preinit.cc)

include_directories(..)

if(ANDROID)
  include_directories(${COMPILER_RT_EXTRA_ANDROID_HEADERS})
endif()

set(ASAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_no_rtti_flag(ASAN_CFLAGS)

set(ASAN_COMMON_DEFINITIONS
  ASAN_HAS_EXCEPTIONS=1)

if(ANDROID)
  list(APPEND ASAN_COMMON_DEFINITIONS
    ASAN_LOW_MEMORY=1)
endif()

set(ASAN_DYNAMIC_DEFINITIONS
  ${ASAN_COMMON_DEFINITIONS} ASAN_DYNAMIC=1)

set(ASAN_DYNAMIC_CFLAGS ${ASAN_CFLAGS})
append_if(COMPILER_RT_HAS_FTLS_MODEL_INITIAL_EXEC
  -ftls-model=initial-exec ASAN_DYNAMIC_CFLAGS)

set(ASAN_DYNAMIC_LIBS stdc++ m c)
append_if(COMPILER_RT_HAS_LIBPTHREAD pthread ASAN_DYNAMIC_LIBS)
append_if(COMPILER_RT_HAS_LIBDL dl ASAN_DYNAMIC_LIBS)

# Compile ASan sources into an object library.
if(APPLE)
  foreach(os ${SANITIZER_COMMON_SUPPORTED_DARWIN_OS})
    add_compiler_rt_darwin_object_library(RTAsan ${os}
      ARCH ${ASAN_SUPPORTED_ARCH}
      SOURCES ${ASAN_SOURCES} ${ASAN_CXX_SOURCES}
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
  endforeach()
elseif(ANDROID)
  add_library(RTAsan.arm.android OBJECT ${ASAN_SOURCES} ${ASAN_CXX_SOURCES})
  set_target_compile_flags(RTAsan.arm.android ${ASAN_CFLAGS})
  set_property(TARGET RTAsan.arm.android APPEND PROPERTY
    COMPILE_DEFINITIONS ${ASAN_COMMON_DEFINITIONS})
else()
  foreach(arch ${ASAN_SUPPORTED_ARCH})
    add_compiler_rt_object_library(RTAsan ${arch}
      SOURCES ${ASAN_SOURCES} CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    add_compiler_rt_object_library(RTAsan_cxx ${arch}
      SOURCES ${ASAN_CXX_SOURCES} CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    add_compiler_rt_object_library(RTAsan_preinit ${arch}
      SOURCES ${ASAN_PREINIT_SOURCES} CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    if (COMPILER_RT_BUILD_SHARED_ASAN)
      add_compiler_rt_object_library(RTAsan_dynamic ${arch}
        SOURCES ${ASAN_SOURCES} ${ASAN_CXX_SOURCES}
        CFLAGS ${ASAN_DYNAMIC_CFLAGS}
        DEFS ${ASAN_DYNAMIC_DEFINITIONS})
    endif()
  endforeach()
endif()

# Build ASan runtimes shipped with Clang.
add_custom_target(asan)
if(APPLE)
  foreach (os ${SANITIZER_COMMON_SUPPORTED_DARWIN_OS})
    add_compiler_rt_darwin_dynamic_runtime(clang_rt.asan_${os}_dynamic ${os}
      ARCH ${ASAN_SUPPORTED_ARCH}
      SOURCES $<TARGET_OBJECTS:RTAsan.${os}>
              $<TARGET_OBJECTS:RTInterception.${os}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${os}>
              $<TARGET_OBJECTS:RTLSanCommon.${os}>
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    add_dependencies(asan clang_rt.asan_${os}_dynamic)
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
  add_dependencies(asan clang_rt.asan-arm-android)
  install(TARGETS clang_rt.asan-arm-android
          ARCHIVE DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR}
          LIBRARY DESTINATION ${COMPILER_RT_LIBRARY_INSTALL_DIR})
else()
  # Build separate libraries for each target.
  foreach(arch ${ASAN_SUPPORTED_ARCH})
    set(ASAN_COMMON_RUNTIME_OBJECTS
      $<TARGET_OBJECTS:RTInterception.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>)
    if (NOT WIN32)
      # We can't build Leak Sanitizer on Windows yet.
      list(APPEND ASAN_COMMON_RUNTIME_OBJECTS
           $<TARGET_OBJECTS:RTLSanCommon.${arch}>)
    endif()

    add_compiler_rt_runtime(clang_rt.asan-${arch} ${arch} STATIC
      SOURCES $<TARGET_OBJECTS:RTAsan_preinit.${arch}>
              $<TARGET_OBJECTS:RTAsan.${arch}>
              ${ASAN_COMMON_RUNTIME_OBJECTS}
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    add_dependencies(asan clang_rt.asan-${arch})

    add_compiler_rt_runtime(clang_rt.asan_cxx-${arch} ${arch} STATIC
      SOURCES $<TARGET_OBJECTS:RTAsan_cxx.${arch}>
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    add_dependencies(asan clang_rt.asan_cxx-${arch})

    if (COMPILER_RT_BUILD_SHARED_ASAN)
      add_compiler_rt_runtime(clang_rt.asan-preinit-${arch} ${arch} STATIC
        SOURCES $<TARGET_OBJECTS:RTAsan_preinit.${arch}>
        CFLAGS ${ASAN_CFLAGS}
        DEFS ${ASAN_COMMON_DEFINITIONS})
      add_dependencies(asan clang_rt.asan-preinit-${arch})

      add_compiler_rt_runtime(clang_rt.asan-dynamic-${arch} ${arch} SHARED
        OUTPUT_NAME clang_rt.asan-${arch}
        SOURCES $<TARGET_OBJECTS:RTAsan_dynamic.${arch}>
                ${ASAN_COMMON_RUNTIME_OBJECTS}
        CFLAGS ${ASAN_DYNAMIC_CFLAGS}
        DEFS ${ASAN_DYNAMIC_DEFINITIONS})
      target_link_libraries(clang_rt.asan-dynamic-${arch} ${ASAN_DYNAMIC_LIBS})
      add_dependencies(asan clang_rt.asan-dynamic-${arch})
    endif()

    if (UNIX AND NOT ${arch} STREQUAL "i386")
      add_sanitizer_rt_symbols(clang_rt.asan_cxx-${arch})
      add_dependencies(asan clang_rt.asan_cxx-${arch}-symbols)
      add_sanitizer_rt_symbols(clang_rt.asan-${arch} asan.syms.extra)
      add_dependencies(asan clang_rt.asan-${arch}-symbols)
    endif()

    if (WIN32)
      add_compiler_rt_runtime(clang_rt.asan_dll_thunk-${arch} ${arch} STATIC
        SOURCES asan_dll_thunk.cc
                $<TARGET_OBJECTS:RTInterception.${arch}>
        CFLAGS ${ASAN_CFLAGS} -DASAN_DLL_THUNK
        DEFS ${ASAN_COMMON_DEFINITIONS})
      add_dependencies(asan clang_rt.asan_dll_thunk-${arch})
    endif()
  endforeach()
endif()

add_compiler_rt_resource_file(asan_blacklist asan_blacklist.txt)
add_dependencies(asan asan_blacklist)
add_dependencies(compiler-rt asan)

add_subdirectory(scripts)

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
