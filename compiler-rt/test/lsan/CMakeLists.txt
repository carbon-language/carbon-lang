set(LSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(LSAN_LIT_TEST_MODE "Standalone")
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/LsanConfig/lit.site.cfg)

set(LSAN_LIT_TEST_MODE "AddressSanitizer")
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/AsanConfig/lit.site.cfg)

set(LSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND LSAN_TEST_DEPS lsan asan)
endif()
add_lit_testsuite(check-lsan "Running the LeakSanitizer tests"
  ${CMAKE_CURRENT_BINARY_DIR}/LsanConfig
  ${CMAKE_CURRENT_BINARY_DIR}/AsanConfig
  DEPENDS ${LSAN_TEST_DEPS})
set_target_properties(check-lsan PROPERTIES FOLDER "LSan tests")
