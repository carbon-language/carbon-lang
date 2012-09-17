configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

# Run TSan unit tests.
if(LLVM_INCLUDE_TESTS)
  add_lit_testsuite(check-tsan "Running ThreadSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}/Unit
    DEPENDS TsanUnitTests)
  set_target_properties(check-tsan PROPERTIES FOLDER "TSan unittests")
endif()
