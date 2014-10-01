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
    # Main UBSan runtime.
    add_compiler_rt_runtime(clang_rt.ubsan-${arch} ${arch} STATIC
      SOURCES ${UBSAN_SOURCES}
      CFLAGS ${UBSAN_CFLAGS})
    # C++-specific parts of UBSan runtime. Requires a C++ ABI library.
    add_compiler_rt_runtime(clang_rt.ubsan_cxx-${arch} ${arch} STATIC
      SOURCES ${UBSAN_CXX_SOURCES}
      CFLAGS ${UBSAN_CXXFLAGS})
    add_dependencies(ubsan
      clang_rt.san-${arch}
      clang_rt.ubsan-${arch}
      clang_rt.ubsan_cxx-${arch})
    if (UNIX AND NOT ${arch} STREQUAL "i386" AND NOT ${arch} STREQUAL "i686")
      add_sanitizer_rt_symbols(clang_rt.ubsan-${arch} ubsan.syms.extra)
      add_sanitizer_rt_symbols(clang_rt.ubsan_cxx-${arch} ubsan.syms.extra)
      add_dependencies(ubsan
        clang_rt.ubsan-${arch}-symbols
        clang_rt.ubsan_cxx-${arch}-symbols)
    endif()
  endforeach()
endif()

add_dependencies(compiler-rt ubsan)
