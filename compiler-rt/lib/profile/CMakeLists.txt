set(PROFILE_SOURCES
  GCDAProfiling.c)

filter_available_targets(PROFILE_SUPPORTED_ARCH x86_64 i386)

if(NOT APPLE)
  # FIXME: Add support for profile.rt on Mac.
  foreach(arch ${PROFILE_SUPPORTED_ARCH})
    add_compiler_rt_static_runtime(clang_rt.profile-${arch} ${arch}
      SOURCES ${PROFILE_SOURCES})
  endforeach()
endif()
