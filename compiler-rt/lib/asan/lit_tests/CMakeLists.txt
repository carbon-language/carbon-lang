set(ASAN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ASAN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/64bitConfig/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/64bitConfig/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/32bitConfig/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  set(ASAN_TESTSUITES)
  if(CAN_TARGET_i386)
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/32bitConfig)
  endif()
  if(CAN_TARGET_x86_64 OR CAN_TARGET_powerpc64)
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/64bitConfig)
  endif()
  # Run ASan tests only if we're sure we may produce working binaries.
  set(ASAN_TEST_DEPS
    ${SANITIZER_COMMON_LIT_TEST_DEPS}
    ${ASAN_RUNTIME_LIBRARIES}
    asan_blacklist)
  set(ASAN_TEST_PARAMS
    asan_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)
  if(LLVM_INCLUDE_TESTS)
    list(APPEND ASAN_TEST_DEPS AsanUnitTests)
    list(APPEND ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit)
  endif()
  add_lit_testsuite(check-asan "Running the AddressSanitizer tests"
    ${ASAN_TESTSUITES}
    PARAMS ${ASAN_TEST_PARAMS}
    DEPENDS ${ASAN_TEST_DEPS})
  set_target_properties(check-asan PROPERTIES FOLDER "ASan tests")
endif()
