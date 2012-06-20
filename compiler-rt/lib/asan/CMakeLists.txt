# Build for the AddressSanitizer runtime support library.

include_directories(..)

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

if(CAN_TARGET_X86_64)
  add_library(clang_rt.asan-x86_64 STATIC ${ASAN_SOURCES})
  set_target_properties(clang_rt.asan-x86_64 PROPERTIES COMPILE_FLAGS "${TARGET_X86_64_CFLAGS}")
endif()
if(CAN_TARGET_I386)
  add_library(clang_rt.asan-i386 STATIC ${ASAN_SOURCES})
  set_target_properties(clang_rt.asan-i386 PROPERTIES COMPILE_FLAGS "${TARGET_I386_CFLAGS}")
endif()
