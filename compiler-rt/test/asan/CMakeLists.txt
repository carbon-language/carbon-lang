set(ASAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(ASAN_TESTSUITES)

macro(get_bits_for_arch arch bits)
  if (${arch} STREQUAL "arm" OR
      ${arch} STREQUAL "i386" OR
      ${arch} STREQUAL "i686" OR
      ${arch} STREQUAL "mips")
    set(bits 32)
  elseif (${arch} STREQUAL "aarch64" OR
      ${arch} STREQUAL "x86_64" OR
      ${arch} STREQUAL "mips64")
    set(bits 64)
  else()
    message(FATAL_ERROR "Unknown target architecture: ${arch}")
  endif()
endmacro()

# TODO: merge with non-ANDROID case
if(ANDROID)
  foreach(arch ${ASAN_SUPPORTED_ARCH})
    set(ASAN_TEST_TARGET_CC ${CMAKE_CXX_COMPILER})
    set(ASAN_TEST_TARGET_CFLAGS ${COMPILER_RT_TEST_COMPILER_CFLAGS})
    set(ASAN_TEST_CONFIG_SUFFIX "-${arch}-android")
    get_bits_for_arch(${arch} ASAN_TEST_BITS)
    set(ASAN_TEST_DYNAMIC True)
    set(ASAN_TEST_TARGET_ARCH "${arch}-android")
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG ${ARCH_UPPER_CASE}AndroidConfig)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG}/lit.site.cfg
      )
    list(APPEND ASAN_TESTSUITES
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG})
  endforeach()

else()  # Not Android

  if(CAN_TARGET_arm)
    # This is only true if we are cross-compiling.
    # Build all tests with host compiler and use host tools.
    set(ASAN_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
    set(ASAN_TEST_TARGET_CFLAGS ${COMPILER_RT_TEST_COMPILER_CFLAGS})
    set(ASAN_TEST_CONFIG_SUFFIX "-arm-linux")
    set(ASAN_TEST_BITS "32")
    set(ASAN_TEST_DYNAMIC False)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/ARMLinuxConfig/lit.site.cfg
      )
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/ARMLinuxConfig)
  endif()

  if(CAN_TARGET_aarch64)
    # This is only true if we are cross-compiling.
    # Build all tests with host compiler and use host tools.
    set(ASAN_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
    set(ASAN_TEST_TARGET_CFLAGS ${COMPILER_RT_TEST_COMPILER_CFLAGS})
    set(ASAN_TEST_CONFIG_SUFFIX "-aarch64-linux")
    set(ASAN_TEST_BITS "64")
    set(ASAN_TEST_DYNAMIC False)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/AArch64LinuxConfig/lit.site.cfg
      )
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/AArch64LinuxConfig)
  endif()

  if(CAN_TARGET_x86_64 OR CAN_TARGET_powerpc64)
    set(ASAN_TEST_CONFIG_SUFFIX "64")
    set(ASAN_TEST_BITS "64")
    set(ASAN_TEST_TARGET_CFLAGS ${TARGET_64_BIT_CFLAGS})
    set(ASAN_TEST_DYNAMIC False)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/64bitConfig/lit.site.cfg
      )
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/64bitConfig)
    if(COMPILER_RT_BUILD_SHARED_ASAN)
      set(ASAN_TEST_CONFIG_SUFFIX "64-Dynamic")
      set(ASAN_TEST_DYNAMIC True)
      configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
        ${CMAKE_CURRENT_BINARY_DIR}/64bitConfig-dynamic/lit.site.cfg)
      list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/64bitConfig-dynamic)
    endif()
  endif()

  if(CAN_TARGET_i386)
    set(ASAN_TEST_CONFIG_SUFFIX "32")
    set(ASAN_TEST_BITS "32")
    set(ASAN_TEST_TARGET_CFLAGS ${TARGET_32_BIT_CFLAGS})
    set(ASAN_TEST_DYNAMIC False)
    set(ASAN_TEST_TARGET_ARCH "i386")
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig/lit.site.cfg
      )
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig)
    if(COMPILER_RT_BUILD_SHARED_ASAN)
      set(ASAN_TEST_CONFIG_SUFFIX "32-Dynamic")
      set(ASAN_TEST_DYNAMIC True)
      configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
        ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig-dynamic/lit.site.cfg)
      list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig-dynamic)
    endif()
  endif()

  if(CAN_TARGET_mips OR CAN_TARGET_mipsel)
    set(ASAN_TEST_CONFIG_SUFFIX "32")
    set(ASAN_TEST_BITS "32")
    set(ASAN_TEST_TARGET_CFLAGS ${TARGET_32_BIT_CFLAGS})
    set(ASAN_TEST_DYNAMIC False)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig/lit.site.cfg
      )
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig)
    if(COMPILER_RT_BUILD_SHARED_ASAN)
      set(ASAN_TEST_CONFIG_SUFFIX "32-Dynamic")
      set(ASAN_TEST_DYNAMIC True)
      configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
        ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig-dynamic/lit.site.cfg)
      list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig-dynamic)
    endif()
  endif()
endif()  # Not Android

if(COMPILER_RT_INCLUDE_TESTS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
endif()

set(ASAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(COMPILER_RT_STANDALONE_BUILD)
  list(APPEND ASAN_TEST_DEPS ${LLVM_TOOLS_BINARY_DIR}/FileCheck)
else()
  list(APPEND ASAN_TEST_DEPS asan)
endif()

# FIXME: support unit test in the android test runner
if(COMPILER_RT_INCLUDE_TESTS AND NOT ANDROID)
  list(APPEND ASAN_TEST_DEPS AsanUnitTests)
  list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
endif()
add_lit_testsuite(check-asan "Running the AddressSanitizer tests"
  ${ASAN_TESTSUITES}
  DEPENDS ${ASAN_TEST_DEPS})
set_target_properties(check-asan PROPERTIES FOLDER "ASan tests")
