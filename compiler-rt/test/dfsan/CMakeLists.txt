set(DFSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

set(DFSAN_TEST_DEPS
  ${SANITIZER_COMMON_LIT_TEST_DEPS}
  dfsan)
add_lit_testsuite(check-dfsan "Running the DataFlowSanitizer tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DFSAN_TEST_DEPS})
set_target_properties(check-dfsan PROPERTIES FOLDER "DFSan tests")
