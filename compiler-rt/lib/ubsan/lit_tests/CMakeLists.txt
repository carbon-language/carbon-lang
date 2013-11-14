set(UBSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/UbsanConfig/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/UbsanConfig/lit.site.cfg)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/AsanConfig/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/AsanConfig/lit.site.cfg)

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  # Run UBSan output tests only if we're sure that clang would produce
  # working binaries.
  set(UBSAN_TEST_DEPS
    ${SANITIZER_COMMON_LIT_TEST_DEPS}
    ${UBSAN_RUNTIME_LIBRARIES}
    asan_runtime_libraries)
  add_lit_testsuite(check-ubsan "Running UndefinedBehaviorSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}/UbsanConfig
    ${CMAKE_CURRENT_BINARY_DIR}/AsanConfig
    DEPENDS ${UBSAN_TEST_DEPS})
  set_target_properties(check-ubsan PROPERTIES FOLDER "UBSan unittests")
endif()
