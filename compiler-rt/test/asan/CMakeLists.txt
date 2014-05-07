set(ASAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(ASAN_TESTSUITES)

if(CAN_TARGET_arm_android)
  # This is only true if we are cross-compiling.
  # Build all tests with host compiler and use host tools.
  set(ASAN_TEST_TARGET_CC ${CMAKE_C_COMPILER})
  get_filename_component(ASAN_TEST_LLVM_TOOLS_DIR ${CMAKE_C_COMPILER} PATH)
  set(ASAN_TEST_CONFIG_SUFFIX "-arm-android")
  set(ASAN_TEST_BITS "32")
  get_target_flags_for_arch(arm_android ASAN_TEST_TARGET_CFLAGS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/ARMAndroidConfig/lit.site.cfg
    )
  list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/ARMAndroidConfig)
endif()

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

if(COMPILER_RT_INCLUDE_TESTS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
endif()

set(ASAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND ASAN_TEST_DEPS asan)
endif()

# FIXME: support unit test in the android test runner
if(COMPILER_RT_INCLUDE_TESTS AND NOT CAN_TARGET_arm_android)
  list(APPEND ASAN_TEST_DEPS AsanUnitTests)
  list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
endif()
add_lit_testsuite(check-asan "Running the AddressSanitizer tests"
  ${ASAN_TESTSUITES}
  DEPENDS ${ASAN_TEST_DEPS})
set_target_properties(check-asan PROPERTIES FOLDER "ASan tests")
