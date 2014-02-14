configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

set(TSAN_TEST_DEPS
  ${SANITIZER_COMMON_LIT_TEST_DEPS}
  tsan)

if(LLVM_INCLUDE_TESTS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)
  list(APPEND TSAN_TEST_DEPS TsanUnitTests)
endif()

add_lit_testsuite(check-tsan "Running ThreadSanitizer tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TSAN_TEST_DEPS})
set_target_properties(check-tsan PROPERTIES FOLDER "TSan tests")
