set(SANITIZER_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF SANITIZER_CFLAGS)

set(SANCOV_LIBCDEP_SOURCES
  sancov_flags.cc
  sanitizer_coverage_libcdep.cc)

add_compiler_rt_object_libraries(RTSancovLibc
  ${OS_OPTION}
  ARCHS ${SANITIZER_COMMON_SUPPORTED_ARCH}
  SOURCES ${SANCOV_LIBCDEP_SOURCES}
  CFLAGS ${SANITIZER_CFLAGS}
  DEFS ${SANITIZER_COMMON_DEFINITIONS})

include_directories(..)

