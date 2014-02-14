# First, add the subdirectories which contain feature-based runtime libraries
# and several convenience helper libraries.

# Don't build sanitizers in the bootstrap build.
if(LLVM_USE_SANITIZER STREQUAL "")
  # AddressSanitizer is supported on Linux and Mac OS X.
  # 32-bit Windows support is experimental.
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin|Linux")
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
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin|Linux" AND NOT ANDROID)
    # LSan, UBsan and profile can be built on Mac OS and Linux.
    add_subdirectory(lsan)
    add_subdirectory(profile)
    add_subdirectory(ubsan)
  endif()
  if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND NOT ANDROID)
    # ThreadSanitizer and MemorySanitizer are supported on Linux only.
    add_subdirectory(tsan)
    add_subdirectory(msan)
    add_subdirectory(msandr)
    add_subdirectory(dfsan)
  endif()
endif()

add_subdirectory(builtins)

# Generate configs for running lit and unit tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.common.configured.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.common.configured)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.common.unit.configured.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.common.unit.configured)

