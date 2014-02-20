configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.common.configured.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.common.configured)

# BlocksRuntime and builtins testsuites are not yet ported to lit.
# add_subdirectory(BlocksRuntime)
# add_subdirectory(builtins)

set(SANITIZER_COMMON_LIT_TEST_DEPS)
# When ANDROID, we build tests with the host compiler (i.e. CMAKE_C_COMPILER),
# and run tests with tools from the host toolchain.
if(NOT ANDROID)
  if(NOT COMPILER_RT_STANDALONE_BUILD)
    # Use LLVM utils and Clang from the same build tree.
    list(APPEND SANITIZER_COMMON_LIT_TEST_DEPS
      clang clang-headers FileCheck count not llvm-nm llvm-symbolizer
      compiler-rt-headers)
  endif()
  if(UNIX)
    list(APPEND SANITIZER_COMMON_LIT_TEST_DEPS SanitizerLintCheck)
  endif()
endif()

# Run sanitizer tests only if we're sure that clang would produce
# working binaries.
if(COMPILER_RT_CAN_EXECUTE_TESTS)
  if(ASAN_SUPPORTED_ARCH OR ANDROID)
    add_subdirectory(asan)
  endif()
  if(DFSAN_SUPPORTED_ARCH)
    add_subdirectory(dfsan)
  endif()
  if(LSAN_SUPPORTED_ARCH)
    add_subdirectory(lsan)
  endif()
  if(MSAN_SUPPORTED_ARCH)
    add_subdirectory(msan)
  endif()
  if(SANITIZER_COMMON_SUPPORTED_ARCH)
    add_subdirectory(sanitizer_common)
  endif()
  if(TSAN_SUPPORTED_ARCH)
    add_subdirectory(tsan)
  endif()
  if(UBSAN_SUPPORTED_ARCH)
    add_subdirectory(ubsan)
  endif()
endif()

if(COMPILER_RT_STANDALONE_BUILD)
  # Now that we've traversed all the directories and know all the lit testsuites,
  # introduce a rule to run to run all of them.
  get_property(LLVM_LIT_TESTSUITES GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
  get_property(LLVM_LIT_DEPENDS GLOBAL PROPERTY LLVM_LIT_DEPENDS)
  add_lit_target(check-all
    "Running all regression tests"
    ${LLVM_LIT_TESTSUITES}
    DEPENDS ${LLVM_LIT_DEPENDS})
endif()
