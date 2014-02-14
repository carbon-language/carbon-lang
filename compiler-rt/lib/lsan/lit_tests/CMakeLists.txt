set(LSAN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LSAN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

set(LSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/LsanConfig/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/LsanConfig/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/AsanConfig/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/AsanConfig/lit.site.cfg
  )

if(COMPILER_RT_CAN_EXECUTE_TESTS AND NOT APPLE AND NOT ANDROID)
  set(LSAN_TEST_DEPS
    ${SANITIZER_COMMON_LIT_TEST_DEPS}
    ${LSAN_RUNTIME_LIBRARIES})
  foreach(arch ${LSAN_SUPPORTED_ARCH})
    list(APPEND LSAN_TEST_DEPS clang_rt.asan-${arch})
  endforeach()
  add_lit_testsuite(check-lsan "Running the LeakSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}/LsanConfig
    ${CMAKE_CURRENT_BINARY_DIR}/AsanConfig
    DEPENDS ${LSAN_TEST_DEPS})
  set_target_properties(check-lsan PROPERTIES FOLDER "LSan tests")
endif()
