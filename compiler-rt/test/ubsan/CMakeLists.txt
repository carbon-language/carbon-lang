set(UBSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(UBSAN_LIT_TEST_MODE "Standalone")
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/UbsanConfig/lit.site.cfg)

set(UBSAN_LIT_TEST_MODE "AddressSanitizer")
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/AsanConfig/lit.site.cfg)

set(UBSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND UBSAN_TEST_DEPS ubsan asan)
endif()

add_lit_testsuite(check-ubsan "Running UndefinedBehaviorSanitizer tests"
  ${CMAKE_CURRENT_BINARY_DIR}/UbsanConfig
  ${CMAKE_CURRENT_BINARY_DIR}/AsanConfig
  DEPENDS ${UBSAN_TEST_DEPS})
set_target_properties(check-ubsan PROPERTIES FOLDER "UBSan unittests")
