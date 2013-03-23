# Build for the undefined behavior sanitizer runtime support library.

set(UBSAN_SOURCES
  ubsan_diag.cc
  ubsan_handlers.cc
  ubsan_value.cc
  )

set(UBSAN_CXX_SOURCES
  ubsan_handlers_cxx.cc
  ubsan_type_hash.cc
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
    SOURCES ${UBSAN_SOURCES} ${UBSAN_CXX_SOURCES}
            $<TARGET_OBJECTS:RTSanitizerCommon.osx>
    CFLAGS ${UBSAN_CFLAGS})
  list(APPEND UBSAN_RUNTIME_LIBRARIES clang_rt.ubsan_osx)
else()
  # Build separate libraries for each target.
  foreach(arch ${UBSAN_SUPPORTED_ARCH})
    # Main UBSan runtime.
    add_compiler_rt_static_runtime(clang_rt.ubsan-${arch} ${arch}
      SOURCES ${UBSAN_SOURCES}
      CFLAGS ${UBSAN_CFLAGS}
      SYMS ubsan.syms)
    # C++-specific parts of UBSan runtime. Requires a C++ ABI library.
    add_compiler_rt_static_runtime(clang_rt.ubsan_cxx-${arch} ${arch}
      SOURCES ${UBSAN_CXX_SOURCES}
      CFLAGS ${UBSAN_CFLAGS}
      SYMS ubsan.syms)
    list(APPEND UBSAN_RUNTIME_LIBRARIES
           clang_rt.san-${arch}
           clang_rt.ubsan-${arch}
           clang_rt.ubsan_cxx-${arch})
  endforeach()
endif()

add_subdirectory(lit_tests)
