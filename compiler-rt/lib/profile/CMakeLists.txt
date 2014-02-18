set(PROFILE_SOURCES
  GCDAProfiling.c
  PGOProfiling.c)

filter_available_targets(PROFILE_SUPPORTED_ARCH x86_64 i386 arm)

add_custom_target(profile)

if(APPLE)
  add_compiler_rt_osx_static_runtime(clang_rt.profile_osx
    ARCH ${PROFILE_SUPPORTED_ARCH}
    SOURCES ${PROFILE_SOURCES})
  add_dependencies(profile clang_rt.profile_osx)
else()
  foreach(arch ${PROFILE_SUPPORTED_ARCH})
    add_compiler_rt_static_runtime(clang_rt.profile-${arch}
      ${arch}
      SOURCES ${PROFILE_SOURCES})
    add_dependencies(profile clang_rt.profile-${arch})
  endforeach()
endif()

add_dependencies(compiler-rt profile)
