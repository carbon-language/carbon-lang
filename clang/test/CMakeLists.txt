find_file(CLANG_TEST_RUNNER TestRunner.sh PATHS ${CMAKE_CURRENT_SOURCE_DIR}
  DOC "Clang's regression testing script")
mark_as_advanced(CLANG_TEST_RUNNER)

set(CLANG_TEST_DIRECTORIES
  "Analysis"
  "CodeGen"
  "CodeGenCXX"
  "CodeGenObjC"
  "Coverage"
  "Driver"
  "FixIt"
  "Frontend"
  "Lexer"
  "Misc"
  "PCH"
  "Parser"
  "Preprocessor"
  "Rewriter"
  "Sema"
  "SemaCXX"
  "SemaObjC"
  "SemaObjCXX"
  "SemaTemplate")

include(FindPythonInterp)
if(PYTHONINTERP_FOUND)
  get_target_property(LLVM_TOOLS_PATH clang RUNTIME_OUTPUT_DIRECTORY)
  set(TESTING_EXTRA_PATHS
      "${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}:${LLVM_SOURCE_DIR}/test/Scripts")
  set(all_testdirs)
  foreach(testdir ${CLANG_TEST_DIRECTORIES})
   add_custom_target(clang-test-${testdir} 
      ${PYTHON_EXECUTABLE} 
      ${LLVM_SOURCE_DIR}/tools/clang/utils/test/MultiTestRunner.py
      -s
      "--path=${TESTING_EXTRA_PATHS}"
      ${CMAKE_CURRENT_SOURCE_DIR}/${testdir}/
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Running Clang regression tests in ${testdir}")

    list(APPEND all_testdirs ${CMAKE_CURRENT_SOURCE_DIR}/${testdir}/)
  endforeach()

  add_custom_target(clang-test
    ${PYTHON_EXECUTABLE} 
    ${LLVM_SOURCE_DIR}/tools/clang/utils/test/MultiTestRunner.py
    "--path=${TESTING_EXTRA_PATHS}"
    ${all_testdirs}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS clang clang-cc
    COMMENT "Running Clang regression tests")
endif()  
