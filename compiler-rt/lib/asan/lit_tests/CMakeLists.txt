set(ASAN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ASAN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

if(COMPILER_RT_CAN_EXECUTE_TESTS)
  # Run ASan tests only if we're sure we may produce working binaries.
  set(ASAN_TEST_DEPS
    clang clang-headers FileCheck count not llvm-nm llvm-symbolizer
    ${ASAN_RUNTIME_LIBRARIES}
    )
  set(ASAN_TEST_PARAMS
    asan_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    )
  if(LLVM_INCLUDE_TESTS)
    list(APPEND ASAN_TEST_DEPS AsanUnitTests)
  endif()
  add_lit_testsuite(check-asan "Running the AddressSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${ASAN_TEST_PARAMS}
    DEPENDS ${ASAN_TEST_DEPS}
    )
  set_target_properties(check-asan PROPERTIES FOLDER "ASan tests")
endif()
