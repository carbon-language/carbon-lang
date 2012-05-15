# Build for the AddressSanitizer runtime support library.

file(GLOB TSAN_SOURCES "*.cc")

if(CAN_TARGET_X86_64)
  add_library(clang_rt.tsan-x86_64 STATIC ${TSAN_SOURCES})
  set_target_properties(clang_rt.tsan-x86_64 PROPERTIES COMPILE_FLAGS "${TARGET_X86_64_CFLAGS}")
endif()
