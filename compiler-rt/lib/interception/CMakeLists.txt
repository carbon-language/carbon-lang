# Build for the runtime interception helper library.

set(INTERCEPTION_SOURCES
  interception_linux.cc
  interception_mac.cc
  interception_win.cc
  interception_type_test.cc
  )

include_directories(..)

if (NOT MSVC)
  set(INTERCEPTION_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    -fno-rtti)
else()
  set(INTERCEPTION_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    /GR- /FS)
endif()

if(APPLE)
  # Build universal binary on APPLE.
  foreach(os ${SANITIZER_COMMON_SUPPORTED_DARWIN_OS})
    add_compiler_rt_darwin_object_library(RTInterception ${os}
      ARCH ${SANITIZER_COMMON_SUPPORTED_ARCH}
      SOURCES ${INTERCEPTION_SOURCES}
      CFLAGS ${INTERCEPTION_CFLAGS})
  endforeach()
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
