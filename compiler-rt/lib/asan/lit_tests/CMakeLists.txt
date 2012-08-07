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

set(ASAN_TEST_DEPS
  clang-rt.asan-x86_64 clang-rt.asan-i386
  )
if(LLVM_INCLUDE_TESTS)
  list(APPEND ASAN_TEST_DEPS AsanUnitTests)
endif()

if("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}")
  # Run ASan output tests only if we're not cross-compiling,
  # and can be sure that clang would produce working binaries.
  list(APPEND ASAN_TEST_DEPS
    clang clang-headers FileCheck count not
    )
  add_lit_testsuite(check-asan "Running the AddressSanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ASAN_TEST_DEPS}
    )
  set_target_properties(check-asan PROPERTIES FOLDER "ASan tests")
elseif(LLVM_INCLUDE_TESTS)
  # Otherwise run only ASan unit tests.
  add_lit_testsuite(check-asan "Running the AddressSanitizer unit tests"
    ${CMAKE_CURRENT_BINARY_DIR}/Unit
    DEPENDS ${ASAN_TEST_DEPS}
    )
  set_target_properties(check-asan PROPERTIES FOLDER "ASan unit tests")
endif()
