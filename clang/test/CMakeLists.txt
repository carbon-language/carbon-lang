set(CLANG_TEST_DIRECTORIES
  "Analysis"
  "CodeCompletion"
  "CodeGen"
  "CodeGenCXX"
  "CodeGenObjC"
  "CodeGenOpenCL"
  "Coverage"
  "CXX"
  "Driver"
  "FixIt"
  "Frontend"
  "Headers"
  "Index"
  "Lexer"
  "Misc"
  "PCH"
  "Parser"
  "Preprocessor"
  "Rewriter"
  "Sema"
  "SemaCUDA"
  "SemaCXX"
  "SemaObjC"
  "SemaObjCXX"
  "SemaOpenCL"
  "SemaTemplate")

set(LLVM_SOURCE_DIR "${LLVM_MAIN_SRC_DIR}")
set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
set(LLVM_BUILD_MODE "%(build_mode)s")
set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}/%(build_config)s")
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib/%(build_config)s")
set(CLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CLANG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")
if(BUILD_SHARED_LIBS)
  set(ENABLE_SHARED 1)
else()
  set(ENABLE_SHARED 0)
endif(BUILD_SHARED_LIBS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  @ONLY)

include(FindPythonInterp)
if(PYTHONINTERP_FOUND)
  if( LLVM_MAIN_SRC_DIR )
    set(LIT "${LLVM_SOURCE_DIR}/utils/lit/lit.py")
  else()
    set(LIT "${PATH_TO_LLVM_BUILD}/bin/${CMAKE_CFG_INTDIR}/llvm-lit")
  endif()

  if( PATH_TO_LLVM_BUILD )
    set(CLANG_TEST_EXTRA_ARGS "--path=${CLANG_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}")
  endif()

  option(CLANG_TEST_USE_VG "Run Clang tests under Valgrind" OFF)
  if(CLANG_TEST_USE_VG)
    set(CLANG_TEST_EXTRA_ARGS ${CLANG_TEST_EXTRA_ARGS} "--vg")
  endif ()

  set(LIT_ARGS "${CLANG_TEST_EXTRA_ARGS} ${LLVM_LIT_ARGS}")
  separate_arguments(LIT_ARGS)

  add_custom_target(clang-test.deps)

  foreach(testdir ${CLANG_TEST_DIRECTORIES})
    add_custom_target(clang-test-${testdir}
      COMMAND ${PYTHON_EXECUTABLE}
                  ${LIT}
                  --param clang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                  --param build_config=${CMAKE_CFG_INTDIR}
                  ${LIT_ARGS}
                  ${CMAKE_CURRENT_BINARY_DIR}/${testdir}
                  DEPENDS clang c-index-test FileCheck not count
                  COMMENT "Running Clang regression tests in ${testdir}")
  endforeach()

  add_custom_target(clang-test
    COMMAND ${PYTHON_EXECUTABLE}
                ${LIT}
                --param clang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                --param clang_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
                --param build_config=${CMAKE_CFG_INTDIR}
                --param build_mode=${RUNTIME_BUILD_MODE}
                ${LIT_ARGS}
                ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Running Clang regression tests")

  add_custom_target(clang-c++tests
    COMMAND ${PYTHON_EXECUTABLE}
                ${LIT}
                --param clang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                --param build_config=${CMAKE_CFG_INTDIR}
                ${LIT_ARGS}
                ${CMAKE_CURRENT_SOURCE_DIR}/../utils/C++Tests
                DEPENDS clang c-index-test FileCheck not count
                COMMENT "Running Clang regression tests")

  if( NOT CLANG_BUILT_STANDALONE )
     add_custom_target(check-all
      COMMAND ${PYTHON_EXECUTABLE}
                ${LIT}
                --param build_config=${CMAKE_CFG_INTDIR}
                --param build_mode=${RUNTIME_BUILD_MODE}
                ${LIT_ARGS}
                ${LLVM_BINARY_DIR}/test
                ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Running Clang and LLVM regression tests")
    add_dependencies(check-all check.deps clang-test.deps)
    add_dependencies(clang-test.deps
                     ClangUnitTests
                     llvm-dis opt
                     FileCheck count not
                     )

 endif()

  add_dependencies(clang-test clang-test.deps)
  add_dependencies(clang-test.deps
                 clang clang-headers c-index-test
                 )

endif()
