if(COMPILER_RT_INCLUDE_TESTS)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)
  add_lit_testsuite(check-sanitizer "Running sanitizer library unittests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS SanitizerUnitTests)
  set_target_properties(check-sanitizer PROPERTIES FOLDER "Sanitizer unittests")
endif()
