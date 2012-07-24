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

set(ASAN_COMMON_DEFINITIONS
  ASAN_HAS_EXCEPTIONS=1
  ASAN_NEEDS_SEGV=1
  )

# Compute the Clang version from the LLVM version.
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
       ${PACKAGE_VERSION})

# Set up paths to ASan runtime that Clang driver passes to linker.
if (APPLE)
  # FIXME: We need to build OS X universal binary libclang_rt.asan-osx.a
  # instead of two arch-specific binaries.
  set(clang_lib_dir ${LLVM_BINARY_DIR}/lib/clang/${CLANG_VERSION}/lib/darwin)
  list(APPEND ASAN_CFLAGS -mmacosx-version-min=10.5)
elseif (UNIX)
  # Assume Linux.
  set(clang_lib_dir ${LLVM_BINARY_DIR}/lib/clang/${CLANG_VERSION}/lib/linux)
endif()

if(CAN_TARGET_X86_64)
  add_library(clang_rt.asan-x86_64 STATIC
    ${ASAN_SOURCES}
    $<TARGET_OBJECTS:RTInterception.x86_64>
    $<TARGET_OBJECTS:RTSanitizerCommon.x86_64>
    )
  set_target_compile_flags(clang_rt.asan-x86_64
    ${ASAN_CFLAGS}
    ${TARGET_X86_64_CFLAGS}
    )
  set_property(TARGET clang_rt.asan-x86_64 APPEND PROPERTY COMPILE_DEFINITIONS
    ${ASAN_COMMON_DEFINITIONS})
  add_custom_command (TARGET clang_rt.asan-x86_64 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:clang_rt.asan-x86_64>
                                        ${clang_lib_dir}/libclang_rt.asan-x86_64.a
    COMMENT "Copying libclang_rt.asan-x86_64.a to Clang lib dir...")
endif()
if(CAN_TARGET_I386)
  add_library(clang_rt.asan-i386 STATIC
    ${ASAN_SOURCES}
    $<TARGET_OBJECTS:RTInterception.i386>
    $<TARGET_OBJECTS:RTSanitizerCommon.i386>
    )
  set_target_compile_flags(clang_rt.asan-i386
    ${ASAN_CFLAGS}
    ${TARGET_I386_CFLAGS}
    )
  set_property(TARGET clang_rt.asan-i386 APPEND PROPERTY COMPILE_DEFINITIONS
    ${ASAN_COMMON_DEFINITIONS})
  add_custom_command (TARGET clang_rt.asan-i386 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:clang_rt.asan-i386>
                                        ${clang_lib_dir}/libclang_rt.asan-i386.a
    COMMENT "Copying libclang_rt.asan-i386.a to Clang lib dir...")
endif()

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
