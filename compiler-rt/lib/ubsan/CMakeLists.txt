# Build for the undefined behavior sanitizer runtime support library.

set(UBSAN_SOURCES
  ubsan_diag.cc
  ubsan_init.cc
  ubsan_flags.cc
  ubsan_handlers.cc
  ubsan_value.cc
  )

set(UBSAN_CXX_SOURCES
  ubsan_handlers_cxx.cc
  ubsan_type_hash.cc
  )

include_directories(..)

set(UBSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_no_rtti_flag(UBSAN_CFLAGS)
set(UBSAN_CXXFLAGS ${SANITIZER_COMMON_CFLAGS})

add_custom_target(ubsan)

if(APPLE)
  # Build universal binary on APPLE.
  add_compiler_rt_osx_static_runtime(clang_rt.ubsan_osx
    ARCH ${UBSAN_SUPPORTED_ARCH}
    SOURCES ${UBSAN_SOURCES} ${UBSAN_CXX_SOURCES}
            $<TARGET_OBJECTS:RTSanitizerCommon.osx>
    CFLAGS ${UBSAN_CXXFLAGS})
  add_dependencies(ubsan clang_rt.ubsan_osx)
else()
  # Build separate libraries for each target.
  foreach(arch ${UBSAN_SUPPORTED_ARCH})
    add_compiler_rt_object_library(RTUbsan ${arch}
      SOURCES ${UBSAN_SOURCES} CFLAGS ${UBSAN_CFLAGS})
    # C++-specific parts of UBSan runtime. Requires a C++ ABI library.
    add_compiler_rt_object_library(RTUbsan_cxx ${arch}
      SOURCES ${UBSAN_CXX_SOURCES} CFLAGS ${UBSAN_CXXFLAGS})

    # Standalone UBSan runtimes.
    add_compiler_rt_runtime(clang_rt.ubsan_standalone-${arch} ${arch} STATIC
      SOURCES $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
              $<TARGET_OBJECTS:RTUbsan.${arch}>
      CFLAGS ${UBSAN_CFLAGS})
    add_compiler_rt_runtime(clang_rt.ubsan_standalone_cxx-${arch} ${arch} STATIC
      SOURCES $<TARGET_OBJECTS:RTUbsan_cxx.${arch}>
      CFLAGS ${UBSAN_CXXFLAGS})
    # UBSan runtimes used when another sanitizer is available.
    add_compiler_rt_runtime(clang_rt.ubsan-${arch} ${arch} STATIC
      SOURCES $<TARGET_OBJECTS:RTUbsan.${arch}>
      CFLAGS ${UBSAN_CFLAGS})
    add_compiler_rt_runtime(clang_rt.ubsan_cxx-${arch} ${arch} STATIC
      SOURCES $<TARGET_OBJECTS:RTUbsan_cxx.${arch}>
      CFLAGS ${UBSAN_CXXFLAGS})

    add_dependencies(ubsan
      clang_rt.ubsan-${arch}
      clang_rt.ubsan_cxx-${arch}
      clang_rt.ubsan_standalone-${arch}
      clang_rt.ubsan_standalone_cxx-${arch})
    if (UNIX AND NOT ${arch} MATCHES "i386|i686")
      add_sanitizer_rt_symbols(clang_rt.ubsan-${arch} ubsan.syms.extra)
      add_sanitizer_rt_symbols(clang_rt.ubsan_cxx-${arch} ubsan.syms.extra)
      add_sanitizer_rt_symbols(clang_rt.ubsan_standalone-${arch} ubsan.syms.extra)
      add_sanitizer_rt_symbols(clang_rt.ubsan_standalone_cxx-${arch} ubsan.syms.extra)
      add_dependencies(ubsan
        clang_rt.ubsan-${arch}-symbols
        clang_rt.ubsan_cxx-${arch}-symbols
        clang_rt.ubsan_standalone-${arch}-symbols
        clang_rt.ubsan_standalone_cxx-${arch}-symbols)
    endif()
  endforeach()
endif()

add_dependencies(compiler-rt ubsan)
