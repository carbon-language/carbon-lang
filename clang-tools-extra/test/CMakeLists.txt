# Test runner infrastructure for Clang-based tools. This configures the Clang
# test trees for use by Lit, and delegates to LLVM's lit test handlers.
#
# Note that currently we don't support stand-alone builds of Clang, you must
# be building Clang from within a combined LLVM+Clang checkout..

set(CLANG_TOOLS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CLANG_TOOLS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")

option(CLANG_TOOLS_TEST_USE_VG "Run Clang tools' tests under Valgrind" OFF)
if(CLANG_TOOLS_TEST_USE_VG)
  set(CLANG_TOOLS_TEST_EXTRA_ARGS ${CLANG_TEST_EXTRA_ARGS} "--vg")
endif()

add_subdirectory(cpp11-migrate)

set(TEST_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_EXEC_ROOT ${CMAKE_CURRENT_BINARY_DIR})
set(TESTSUITE_NAME "Clang Tools")
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

set(CLANG_TOOLS_TEST_DEPS
  # Base line deps.
  clang clang-headers FileCheck count not

  # cpp11-migrate auto-generated tests. See cpp11-migrate/CMakeLists.txt.
  cpp11-migrate-autogen

  # Individual tools we test.
  remove-cstr-calls cpp11-migrate clang-format
  )

add_lit_testsuite(check-clang-tools "Running the Clang extra tools' regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/cpp11-migrate/generated_tests
  DEPENDS ${CLANG_TOOLS_TEST_DEPS}
  ARGS ${CLANG_TOOLS_TEST_EXTRA_ARGS}
  )
set_target_properties(check-clang-tools PROPERTIES FOLDER "Clang extra tools' tests")
