# First, add the subdirectories which contain feature-based runtime libraries
# and several convenience helper libraries.

include(AddCompilerRT)
include(SanitizerUtils)

if(COMPILER_RT_HAS_SANITIZER_COMMON)
  add_subdirectory(interception)
  add_subdirectory(sanitizer_common)
endif()

if(COMPILER_RT_HAS_ASAN)
  add_subdirectory(asan)
endif()

add_subdirectory(builtins)

if(COMPILER_RT_HAS_DFSAN)
  add_subdirectory(dfsan)
endif()

if(COMPILER_RT_HAS_LSAN OR COMPILER_RT_HAS_LSAN_COMMON)
  add_subdirectory(lsan)
endif()

if(COMPILER_RT_HAS_MSAN)
  add_subdirectory(msan)
endif()

if(COMPILER_RT_HAS_PROFILE)
  add_subdirectory(profile)
endif()

if(COMPILER_RT_HAS_TSAN)
  add_subdirectory(tsan)
  add_subdirectory(tsan/dd)
endif()

if(COMPILER_RT_HAS_UBSAN)
  add_subdirectory(ubsan)
endif()

