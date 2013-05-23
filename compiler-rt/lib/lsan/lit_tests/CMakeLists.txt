set(LSAN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LSAN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  set(LSAN_TEST_DEPS
    ${SANITIZER_COMMON_LIT_TEST_DEPS}
    ${LSAN_RUNTIME_LIBRARIES})
  set(LSAN_TEST_PARAMS
    lsan_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)
  if(LLVM_INCLUDE_TESTS)
    list(APPEND LSAN_TEST_DEPS LsanTests)
  endif()
  add_lit_testsuite(check-lsan "Running the LeakSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${LSAN_TEST_PARAMS}
    DEPENDS ${LSAN_TEST_DEPS})
  set_target_properties(check-lsan PROPERTIES FOLDER "LSan tests")
endif()
