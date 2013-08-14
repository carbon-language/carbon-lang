set(DFSAN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(DFSAN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  # Run DFSan tests only if we're sure we may produce working binaries.
  set(DFSAN_TEST_DEPS
    ${SANITIZER_COMMON_LIT_TEST_DEPS}
    ${DFSAN_RUNTIME_LIBRARIES}
    dfsan_abilist)
  set(DFSAN_TEST_PARAMS
    dfsan_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)
  add_lit_testsuite(check-dfsan "Running the DataFlowSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${DFSAN_TEST_PARAMS}
    DEPENDS ${DFSAN_TEST_DEPS})
  set_target_properties(check-dfsan PROPERTIES FOLDER "DFSan tests")
endif()
