add_custom_target(profile)

set(PROFILE_SOURCES
  GCDAProfiling.c
  InstrProfiling.c
  InstrProfilingBuffer.c
  InstrProfilingFile.c
  InstrProfilingPlatformDarwin.c
  InstrProfilingPlatformLinux.c
  InstrProfilingPlatformOther.c
  InstrProfilingRuntime.cc
  InstrProfilingUtil.c)

if(APPLE)
  add_compiler_rt_runtime(clang_rt.profile
    STATIC
    OS ${PROFILE_SUPPORTED_OS}
    ARCHS ${PROFILE_SUPPORTED_ARCH}
    SOURCES ${PROFILE_SOURCES}
    PARENT_TARGET profile)
else()
  add_compiler_rt_runtime(clang_rt.profile
    STATIC
    ARCHS ${PROFILE_SUPPORTED_ARCH}
    CFLAGS -fPIC
    SOURCES ${PROFILE_SOURCES}
    PARENT_TARGET profile)
endif()

add_dependencies(compiler-rt profile)
