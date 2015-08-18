add_custom_target(profile)

set(PROFILE_SOURCES
  GCDAProfiling.c
  InstrProfiling.c
  InstrProfilingBuffer.c
  InstrProfilingFile.c
  InstrProfilingPlatformDarwin.c
  InstrProfilingPlatformOther.c
  InstrProfilingRuntime.cc
  InstrProfilingUtil.c)

if(APPLE)
  foreach (os ${SANITIZER_COMMON_SUPPORTED_OS})
    add_compiler_rt_darwin_runtime(clang_rt.profile_${os} ${os}
      STATIC
      ARCHS ${PROFILE_SUPPORTED_ARCH}
      SOURCES ${PROFILE_SOURCES})
    add_dependencies(profile clang_rt.profile_${os})
  endforeach()
else()
  foreach(arch ${PROFILE_SUPPORTED_ARCH})
    add_compiler_rt_runtime(clang_rt.profile-${arch} ${arch} STATIC
      CFLAGS -fPIC
      SOURCES ${PROFILE_SOURCES})
    add_dependencies(profile clang_rt.profile-${arch})
  endforeach()
endif()

add_dependencies(compiler-rt profile)
