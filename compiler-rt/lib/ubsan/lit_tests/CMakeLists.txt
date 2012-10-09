configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

if("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}")
  # Run UBSan output tests only if we're not cross-compiling,
  # and can be sure that clang would produce working binaries.
  set(UBSAN_TEST_DEPS
    clang clang-headers FileCheck count not
    ${UBSAN_RUNTIME_LIBRARIES}
    )
  set(UBSAN_TEST_PARAMS
    ubsan_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    )
  add_lit_testsuite(check-ubsan "Running UndefinedBehaviorSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${UBSAN_TEST_PARAMS}
    DEPENDS ${UBSAN_TEST_DEPS}
    )
  set_target_properties(check-ubsan PROPERTIES FOLDER "UBSan unittests")
endif()
