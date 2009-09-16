set(CLANG_TEST_DIRECTORIES
  "Analysis"
  "CodeGen"
  "CodeGenCXX"
  "CodeGenObjC"
  "Coverage"
  "CXX"
  "Driver"
  "FixIt"
  "Frontend"
  "Index"
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
  set(CLANG_TEST_EXTRA_ARGS)
  if (MSVC OR XCODE)
    set(CLANG_TEST_EXTRA_ARGS "--no-progress-bar")
  endif()
  set(all_testdirs)
 foreach(testdir ${CLANG_TEST_DIRECTORIES})
   add_custom_target(clang-test-${testdir} 
      ${PYTHON_EXECUTABLE} 
      ${LLVM_SOURCE_DIR}/tools/clang/utils/test/MultiTestRunner.py
      "--root=${LLVM_SOURCE_DIR}/tools/clang/test"
      "--path=${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}"
      "--path=${LLVM_SOURCE_DIR}/test/Scripts"
      -s ${CLANG_TEST_EXTRA_ARGS}
      --clang=${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}/clang
      --clang-cc=${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}/clang-cc
     ${CMAKE_CURRENT_SOURCE_DIR}/${testdir}/
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Running Clang regression tests in ${testdir}")

    list(APPEND all_testdirs ${CMAKE_CURRENT_SOURCE_DIR}/${testdir}/)
  endforeach()

  add_custom_target(clang-test
    ${PYTHON_EXECUTABLE} 
    ${LLVM_SOURCE_DIR}/tools/clang/utils/test/MultiTestRunner.py
    "--root=${LLVM_SOURCE_DIR}/tools/clang/test"
    "--path=${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}"
    "--path=${LLVM_SOURCE_DIR}/test/Scripts"
    -s ${CLANG_TEST_EXTRA_ARGS}
    --clang=${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}/clang
    --clang-cc=${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}/clang-cc
    ${all_testdirs}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS clang clang-cc index-test
    COMMENT "Running Clang regression tests")
endif()  

# Produce LIT site configuration file.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)
