set(POLLY_TEST_DIRECTORIES
  "ScopInfo"
  "AffineIterator"
  "CodeGen"
  "OpenMP"
  "polybench")

set(LLVM_SOURCE_DIR "${LLVM_MAIN_SRC_DIR}")
set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}/")
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib")
set(POLLY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(POLLY_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

include(FindPythonInterp)
if(PYTHONINTERP_FOUND)
  set(POLLY_TEST_EXTRA_ARGS)
  if (MSVC OR XCODE)
    set(POLLY_TEST_EXTRA_ARGS "--no-progress-bar")
  endif()

  option(POLLY_TEST_USE_VG "Run Polly tests under Valgrind" OFF)
  if(POLLY_TEST_USE_VG)
    set(POLLY_TEST_EXTRA_ARGS ${POLLY_TEST_EXTRA_ARGS} "--vg")
  endif ()

  foreach(testdir ${POLLY_TEST_DIRECTORIES})
    add_custom_target(polly-test-${testdir}
      COMMAND ${PYTHON_EXECUTABLE}
                  ${LLVM_SOURCE_DIR}/utils/lit/lit.py
                  --param polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                  --param build_config=${CMAKE_CFG_INTDIR}
                  -sv ${POLLY_TEST_EXTRA_ARGS}
                  ${CMAKE_CURRENT_BINARY_DIR}/${testdir}
                  DEPENDS opt LLVMPolly
                  COMMENT "Running Polly regression tests in ${testdir}")
  endforeach()

  add_custom_target(polly-test
    COMMAND ${PYTHON_EXECUTABLE}
                ${LLVM_SOURCE_DIR}/utils/lit/lit.py
                --param polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                --param build_config=${CMAKE_CFG_INTDIR}
                -sv ${POLLY_TEST_EXTRA_ARGS}
                ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS opt LLVMPolly
                COMMENT "Running Polly regression tests")
endif()
