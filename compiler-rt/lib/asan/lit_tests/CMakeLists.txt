configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

set(ASAN_TEST_DEPS
  clang clang-headers FileCheck count not
  clang-rt.asan-x86_64 clang-rt.asan-i386
  )

add_lit_testsuite(check-asan "Running the AddressSanitizer tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ASAN_TEST_DEPS}
  )
set_target_properties(check-asan PROPERTIES FOLDER "ASan tests")
