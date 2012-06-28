# Build for the runtime interception helper library.

set(INTERCEPTION_SOURCES
  interception_linux.cc
  interception_mac.cc
  interception_win.cc
  )

# Only add this C file if we're building on a Mac. Other source files can be
# harmlessly compiled on any platform, but the C file is complained about due
# to pedantic rules about empty translation units.
if (APPLE)
  list(APPEND INTERCEPTION_SOURCES mach_override/mach_override.c)
endif ()

set(INTERCEPTION_CFLAGS "-fPIC -fno-exceptions -funwind-tables -fvisibility=hidden")
if (SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  set(INTERCEPTION_CFLAGS "${INTERCEPTION_CFLAGS} -Wno-variadic-macros")
endif ()

set(INTERCEPTION_COMMON_DEFINITIONS
	INTERCEPTION_HAS_EXCEPTIONS=1)

if(CAN_TARGET_X86_64)
  add_library(RTInterception.x86_64 OBJECT ${INTERCEPTION_SOURCES})
  set_property(TARGET RTInterception.x86_64 PROPERTY COMPILE_FLAGS
		"${INTERCEPTION_CFLAGS} ${TARGET_X86_64_CFLAGS}")
  set_property(TARGET RTInterception.x86_64 APPEND PROPERTY COMPILE_DEFINITIONS
		${INTERCEPTION_COMMON_DEFINITIONS})
endif()
if(CAN_TARGET_I386)
  add_library(RTInterception.i386 OBJECT ${INTERCEPTION_SOURCES})
  set_property(TARGET RTInterception.i386 PROPERTY COMPILE_FLAGS
		"${INTERCEPTION_CFLAGS} ${TARGET_I386_CFLAGS}")
  set_property(TARGET RTInterception.i386 APPEND PROPERTY COMPILE_DEFINITIONS
		${INTERCEPTION_COMMON_DEFINITIONS})
endif()
