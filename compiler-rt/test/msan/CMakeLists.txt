set(MSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

set(MSAN_TEST_DEPS
  ${SANITIZER_COMMON_LIT_TEST_DEPS}
  msan)

if(COMPILER_RT_INCLUDE_TESTS AND COMPILER_RT_HAS_LIBCXX_SOURCES)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
  list(APPEND MSAN_TEST_DEPS MsanUnitTests)
endif()

add_lit_testsuite(check-msan "Running the MemorySanitizer tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MSAN_TEST_DEPS}
  )
set_target_properties(check-msan PROPERTIES FOLDER "MSan tests")
