configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(BOLT_TEST_PARAMS
  bolt_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

list(APPEND BOLT_TEST_DEPS
  llvm-config
  FileCheck count not
  llvm-bolt
  perf2bolt
  merge-fdata
  llvm-nm
  llvm-objdump
  llvm-readobj
  yaml2obj
  llvm-strip
  llvm-dwarfdump
  llvm-mc
  )

add_custom_target(bolt-test-depends DEPENDS ${BOLT_TEST_DEPS})
set_target_properties(bolt-test-depends PROPERTIES FOLDER "BOLT tests")

add_lit_testsuite(check-bolt "Running the BOLT regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${BOLT_TEST_PARAMS}
  DEPENDS ${BOLT_TEST_DEPS}
  ARGS ${BOLT_TEST_EXTRA_ARGS}
  )
set_target_properties(check-bolt PROPERTIES FOLDER "BOLT tests")

add_lit_testsuites(BOLT ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${BOLT_TEST_PARAMS}
  DEPENDS ${BOLT_TEST_DEPS}
)
