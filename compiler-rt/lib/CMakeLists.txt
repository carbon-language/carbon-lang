# First, add the subdirectories which contain feature-based runtime libraries
# and several convenience helper libraries.

include(AddCompilerRT)
include(SanitizerUtils)
# Don't build sanitizers in the bootstrap build.
if(NOT LLVM_USE_SANITIZER)
  # AddressSanitizer is supported on Linux, FreeBSD and Mac OS X.
  # 32-bit Windows support is experimental.
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin|Linux|FreeBSD")
    set(SUPPORTS_BUILDING_ASAN TRUE)
  elseif(CMAKE_SYSTEM_NAME MATCHES "Windows"
         AND MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(SUPPORTS_BUILDING_ASAN TRUE)
  else()
    set(SUPPORTS_BUILDING_ASAN FALSE)
  endif()
  if(SUPPORTS_BUILDING_ASAN)
    add_subdirectory(asan)
    add_subdirectory(interception)
    add_subdirectory(sanitizer_common)
  endif()
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin|Linux|FreeBSD" AND NOT ANDROID)
    # LSan, UBsan and profile can be built on Mac OS, FreeBSD and Linux.
    add_subdirectory(lsan)
    add_subdirectory(profile)
    add_subdirectory(ubsan)
  endif()
  if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND NOT ANDROID)
    # ThreadSanitizer and MemorySanitizer are supported on Linux only.
    add_subdirectory(tsan)
    add_subdirectory(tsan/dd)
    add_subdirectory(msan)
    add_subdirectory(msandr)
    add_subdirectory(dfsan)
  endif()
endif()

add_subdirectory(builtins)
