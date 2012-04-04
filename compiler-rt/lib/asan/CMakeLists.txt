# Build for the AddressSanitizer runtime support library.

file(GLOB ASAN_SOURCES "*.cc")

if(CAN_TARGET_X86_64)
  add_library(clang_rt.asan-x86_64 STATIC ${ASAN_SOURCES})
  set_target_properties(clang_rt.asan-x86_64 PROPERTIES COMPILE_FLAGS "${TARGET_X86_64_CFLAGS}")
endif()
if(CAN_TARGET_I386)
  add_library(clang_rt.asan-i386 STATIC ${ASAN_SOURCES})
  set_target_properties(clang_rt.asan-i386 PROPERTIES COMPILE_FLAGS "${TARGET_I386_CFLAGS}")
endif()
