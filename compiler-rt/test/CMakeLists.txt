configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.common.configured.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.common.configured)

# BlocksRuntime and builtins testsuites are not yet ported to lit.
# add_subdirectory(BlocksRuntime)
# add_subdirectory(builtins)

# Run sanitizer tests only if we're sure that clang would produce
# working binaries.
if(COMPILER_RT_CAN_EXECUTE_TESTS)
  if(DFSAN_SUPPORTED_ARCH)
    add_subdirectory(dfsan)
  endif()
  if(LSAN_SUPPORTED_ARCH)
    add_subdirectory(lsan)
  endif()
  if(MSAN_SUPPORTED_ARCH)
    add_subdirectory(msan)
  endif()
  if(UBSAN_SUPPORTED_ARCH)
    add_subdirectory(ubsan)
  endif()
endif()
