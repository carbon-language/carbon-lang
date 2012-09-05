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

set(INTERCEPTION_CFLAGS ${SANITIZER_COMMON_CFLAGS})

if(APPLE)
  # Build universal binary on APPLE.
  add_library(RTInterception.osx OBJECT ${INTERCEPTION_SOURCES})
  set_target_compile_flags(RTInterception.osx ${INTERCEPTION_CFLAGS})
  filter_available_targets(INTERCEPTION_TARGETS x86_64 i386)
  set_target_properties(RTInterception.osx PROPERTIES
    OSX_ARCHITECTURES "${INTERCEPTION_TARGETS}")
else()
  # Otherwise, build separate libraries for each target.
  if(CAN_TARGET_X86_64)
    add_library(RTInterception.x86_64 OBJECT ${INTERCEPTION_SOURCES})
    set_target_compile_flags(RTInterception.x86_64
      ${INTERCEPTION_CFLAGS} ${TARGET_X86_64_CFLAGS})
  endif()
  if(CAN_TARGET_I386)
    add_library(RTInterception.i386 OBJECT ${INTERCEPTION_SOURCES})
    set_target_compile_flags(RTInterception.i386
      ${INTERCEPTION_CFLAGS} ${TARGET_I386_CFLAGS})
  endif()
  if(CAN_TARGET_ARM_ANDROID)
    add_library(RTInterception.arm.android OBJECT ${INTERCEPTION_SOURCES})
    set_target_compile_flags(RTInterception.arm.android
      ${INTERCEPTION_CFLAGS} ${TARGET_ARM_ANDROID_CFLAGS})
  endif()
endif()
