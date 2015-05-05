# Build for the AddressSanitizer runtime support library.

set(ASAN_SOURCES
  asan_allocator.cc
  asan_activation.cc
  asan_debugging.cc
  asan_fake_stack.cc
  asan_flags.cc
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
  asan_suppressions.cc
  asan_thread.cc
  asan_win.cc)

set(ASAN_CXX_SOURCES
  asan_new_delete.cc)

set(ASAN_PREINIT_SOURCES
  asan_preinit.cc)

include_directories(..)

set(ASAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_no_rtti_flag(ASAN_CFLAGS)

set(ASAN_COMMON_DEFINITIONS
  ASAN_HAS_EXCEPTIONS=1)

set(ASAN_DYNAMIC_LINK_FLAGS)

if(ANDROID)
  list(APPEND ASAN_COMMON_DEFINITIONS
    ASAN_LOW_MEMORY=1)
# On Android, -z global does not do what it is documented to do.
# On Android, -z global moves the library ahead in the lookup order,
# placing it right after the LD_PRELOADs. This is used to compensate for the fact
# that Android linker does not look at the dependencies of the main executable
# that aren't dependencies of the current DSO when resolving symbols from said DSO.
# As a net result, this allows running ASan executables without LD_PRELOAD-ing the
# ASan runtime library.
# The above is applicable to L MR1 or newer.
  if (COMPILER_RT_HAS_Z_GLOBAL)
    list(APPEND ASAN_DYNAMIC_LINK_FLAGS -Wl,-z,global)
  endif()
endif()

set(ASAN_DYNAMIC_DEFINITIONS
  ${ASAN_COMMON_DEFINITIONS} ASAN_DYNAMIC=1)
append_list_if(WIN32 INTERCEPTION_DYNAMIC_CRT ASAN_DYNAMIC_DEFINITIONS)

set(ASAN_DYNAMIC_CFLAGS ${ASAN_CFLAGS})
append_list_if(COMPILER_RT_HAS_FTLS_MODEL_INITIAL_EXEC
  -ftls-model=initial-exec ASAN_DYNAMIC_CFLAGS)
append_list_if(MSVC /DEBUG ASAN_DYNAMIC_CFLAGS)

append_list_if(COMPILER_RT_HAS_LIBC c ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBDL dl ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBM m ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread ASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBSTDCXX stdc++ ASAN_DYNAMIC_LIBS)

append_list_if(ANDROID log ASAN_DYNAMIC_LIBS)

# Compile ASan sources into an object library.
if(APPLE)
  foreach(os ${SANITIZER_COMMON_SUPPORTED_DARWIN_OS})
    add_compiler_rt_darwin_object_library(RTAsan ${os}
      ARCH ${ASAN_SUPPORTED_ARCH}
      SOURCES ${ASAN_SOURCES} ${ASAN_CXX_SOURCES}
      CFLAGS ${ASAN_DYNAMIC_CFLAGS}
      DEFS ${ASAN_DYNAMIC_DEFINITIONS})
  endforeach()
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
    add_compiler_rt_object_library(RTAsan_dynamic ${arch}
      SOURCES ${ASAN_SOURCES} ${ASAN_CXX_SOURCES}
      CFLAGS ${ASAN_DYNAMIC_CFLAGS}
      DEFS ${ASAN_DYNAMIC_DEFINITIONS})
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
              $<TARGET_OBJECTS:RTUbsan.${os}>
      CFLAGS ${ASAN_DYNAMIC_CFLAGS}
      DEFS ${ASAN_DYNAMIC_DEFINITIONS})
    add_dependencies(asan clang_rt.asan_${os}_dynamic)
  endforeach()
else()
  # Build separate libraries for each target.
  foreach(arch ${ASAN_SUPPORTED_ARCH})
    set(ASAN_COMMON_RUNTIME_OBJECTS
      $<TARGET_OBJECTS:RTInterception.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
      $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
      $<TARGET_OBJECTS:RTLSanCommon.${arch}>
      $<TARGET_OBJECTS:RTUbsan.${arch}>)

    add_compiler_rt_runtime(clang_rt.asan-${arch} ${arch} STATIC
      SOURCES $<TARGET_OBJECTS:RTAsan_preinit.${arch}>
              $<TARGET_OBJECTS:RTAsan.${arch}>
              ${ASAN_COMMON_RUNTIME_OBJECTS}
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    add_dependencies(asan clang_rt.asan-${arch})

    add_compiler_rt_runtime(clang_rt.asan_cxx-${arch} ${arch} STATIC
      SOURCES $<TARGET_OBJECTS:RTAsan_cxx.${arch}>
              $<TARGET_OBJECTS:RTUbsan_cxx.${arch}>
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    add_dependencies(asan clang_rt.asan_cxx-${arch})

    add_compiler_rt_runtime(clang_rt.asan-preinit-${arch} ${arch} STATIC
      SOURCES $<TARGET_OBJECTS:RTAsan_preinit.${arch}>
      CFLAGS ${ASAN_CFLAGS}
      DEFS ${ASAN_COMMON_DEFINITIONS})
    add_dependencies(asan clang_rt.asan-preinit-${arch})

    if (UNIX AND NOT ${arch} MATCHES "i386|i686")
      add_sanitizer_rt_version_list(clang_rt.asan-dynamic-${arch}
                                    LIB clang_rt.asan-${arch} clang_rt.asan_cxx-${arch}
                                    EXTRA asan.syms.extra)
      add_dependencies(asan clang_rt.asan-dynamic-${arch}-version-list)
      list(APPEND ASAN_DYNAMIC_LINK_FLAGS
           -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/clang_rt.asan-dynamic-${arch}.vers)
    endif()

    if (WIN32)
      set(SHARED_ASAN_NAME clang_rt.asan_dynamic-${arch}${COMPILER_RT_OS_SUFFIX})
    else()
      set(SHARED_ASAN_NAME clang_rt.asan-${arch}${COMPILER_RT_OS_SUFFIX})
    endif()
    add_compiler_rt_runtime(clang_rt.asan-dynamic-${arch} ${arch} SHARED
      OUTPUT_NAME ${SHARED_ASAN_NAME}
      SOURCES $<TARGET_OBJECTS:RTAsan_dynamic.${arch}>
              $<TARGET_OBJECTS:RTUbsan_cxx.${arch}>
              ${ASAN_COMMON_RUNTIME_OBJECTS}
      CFLAGS ${ASAN_DYNAMIC_CFLAGS}
      LINKFLAGS ${ASAN_DYNAMIC_LINK_FLAGS}
      DEFS ${ASAN_DYNAMIC_DEFINITIONS})
    target_link_libraries(clang_rt.asan-dynamic-${arch} ${ASAN_DYNAMIC_LIBS})
    add_dependencies(asan clang_rt.asan-dynamic-${arch})

    if (UNIX AND NOT ${arch} MATCHES "i386|i686")
      add_dependencies(clang_rt.asan-dynamic-${arch} clang_rt.asan-dynamic-${arch}-version-list)
      add_sanitizer_rt_symbols(clang_rt.asan_cxx-${arch})
      add_dependencies(asan clang_rt.asan_cxx-${arch}-symbols)
      add_sanitizer_rt_symbols(clang_rt.asan-${arch} asan.syms.extra)
      add_dependencies(asan clang_rt.asan-${arch}-symbols)
    endif()

    if (WIN32)
      add_compiler_rt_runtime(clang_rt.asan_dll_thunk-${arch} ${arch} STATIC
        SOURCES asan_win_dll_thunk.cc
                $<TARGET_OBJECTS:RTInterception.${arch}>
        CFLAGS ${ASAN_CFLAGS} -DASAN_DLL_THUNK
        DEFS ${ASAN_COMMON_DEFINITIONS})
      add_dependencies(asan clang_rt.asan_dll_thunk-${arch})
      add_compiler_rt_runtime(clang_rt.asan_dynamic_runtime_thunk-${arch} ${arch}
        STATIC
        SOURCES asan_win_dynamic_runtime_thunk.cc
        CFLAGS ${ASAN_CFLAGS} -DASAN_DYNAMIC_RUNTIME_THUNK -Zl
        DEFS ${ASAN_COMMON_DEFINITIONS})
      add_dependencies(asan clang_rt.asan_dynamic_runtime_thunk-${arch})
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
