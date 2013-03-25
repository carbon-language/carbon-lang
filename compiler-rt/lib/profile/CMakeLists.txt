set(PROFILE_SOURCES
  GCDAProfiling.c)

filter_available_targets(PROFILE_SUPPORTED_ARCH x86_64 i386)

if(APPLE)
  add_compiler_rt_osx_static_runtime(clang_rt.profile_osx
    ARCH ${PROFILE_SUPPORTED_ARCH}
    SOURCES ${PROFILE_SOURCES}
    CFLAGS --sysroot=${COMPILER_RT_DARWIN_SDK_SYSROOT})
else()
  foreach(arch ${PROFILE_SUPPORTED_ARCH})
    add_compiler_rt_static_runtime(clang_rt.profile-${arch} ${arch}
      SOURCES ${PROFILE_SOURCES})
  endforeach()
endif()
