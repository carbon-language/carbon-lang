# Build for the runtime interception helper library.

set(INTERCEPTION_SOURCES
  interception_linux.cc
  interception_mac.cc
  interception_win.cc
  )

set(MACH_OVERRIDE_SOURCES
  mach_override/mach_override.c
  )

include_directories(..)

# Only add this C file if we're building on a Mac. Other source files can be
# harmlessly compiled on any platform, but the C file is complained about due
# to pedantic rules about empty translation units.
if (APPLE)
  list(APPEND INTERCEPTION_SOURCES ${MACH_OVERRIDE_SOURCES})
  set_source_files_properties(${MACH_OVERRIDE_SOURCES} PROPERTIES COMPILE_FLAGS "-std=c99 ${INTERCEPTION_CFLAGS}")
endif ()

set(INTERCEPTION_CFLAGS ${SANITIZER_COMMON_CFLAGS})

if(APPLE)
  # Build universal binary on APPLE.
  add_compiler_rt_osx_object_library(RTInterception
    ARCH ${SANITIZER_COMMON_SUPPORTED_ARCH}
    SOURCES ${INTERCEPTION_SOURCES}
    CFLAGS ${INTERCEPTION_CFLAGS})
elseif(ANDROID)
  add_library(RTInterception.arm.android OBJECT ${INTERCEPTION_SOURCES})
  set_target_compile_flags(RTInterception.arm.android
    ${INTERCEPTION_CFLAGS})
else()
  # Otherwise, build separate libraries for each target.
  foreach(arch ${SANITIZER_COMMON_SUPPORTED_ARCH})
    add_compiler_rt_object_library(RTInterception ${arch}
      SOURCES ${INTERCEPTION_SOURCES} CFLAGS ${INTERCEPTION_CFLAGS})
  endforeach()
endif()
