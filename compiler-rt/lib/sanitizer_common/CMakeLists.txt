# Build system for the common Sanitizer runtime support library components.
# These components are shared between AddressSanitizer and ThreadSanitizer.

set(SANITIZER_SOURCES
  sanitizer_allocator.cc
  sanitizer_common.cc
  sanitizer_libc.cc
  sanitizer_linux.cc
  sanitizer_mac.cc
  sanitizer_posix.cc
  sanitizer_printf.cc
  sanitizer_symbolizer.cc
  sanitizer_win.cc
  )

set(SANITIZER_CFLAGS "-fPIC -fno-exceptions -funwind-tables -fvisibility=hidden")

set(SANITIZER_COMMON_DEFINITIONS
	SANITIZER_HAS_EXCEPTIONS=1)

if(CAN_TARGET_X86_64)
  add_library(RTSanitizerCommon.x86_64 OBJECT ${SANITIZER_SOURCES})
  set_property(TARGET RTSanitizerCommon.x86_64 PROPERTY COMPILE_FLAGS
		"${SANITIZER_CFLAGS} ${TARGET_X86_64_CFLAGS}")
  set_property(TARGET RTSanitizerCommon.x86_64 APPEND PROPERTY COMPILE_DEFINITIONS
		${SANITIZER_COMMON_DEFINITIONS})
endif()
if(CAN_TARGET_I386)
  add_library(RTSanitizerCommon.i386 OBJECT ${SANITIZER_SOURCES})
  set_property(TARGET RTSanitizerCommon.i386 PROPERTY COMPILE_FLAGS
		"${SANITIZER_CFLAGS} ${TARGET_I386_CFLAGS}")
  set_property(TARGET RTSanitizerCommon.i386 APPEND PROPERTY COMPILE_DEFINITIONS
		${SANITIZER_COMMON_DEFINITIONS})
endif()
