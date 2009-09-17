set(CLANG_TEST_DIRECTORIES
  "Analysis"
  "CodeCompletion"
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

  foreach(testdir ${CLANG_TEST_DIRECTORIES})
    add_custom_target(clang-test-${testdir} 
      COMMAND sed -e "s#\@CLANG_SOURCE_DIR\@#${CMAKE_CURRENT_SOURCE_DIR}/..#"
                  -e "s#\@CLANG_BINARY_DIR\@#${CMAKE_CURRENT_BINARY_DIR}/..#"
                  -e "s#\@LLVM_TOOLS_DIR\@#${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}#"
                  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in >
                  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
      COMMAND ${PYTHON_EXECUTABLE} 
                  ${LLVM_SOURCE_DIR}/utils/lit/lit.py
                  -s ${CLANG_TEST_EXTRA_ARGS}
                  ${CMAKE_CURRENT_BINARY_DIR}/${testdir}
                  DEPENDS clang clang-cc index-test
                  COMMENT "Running Clang regression tests in ${testdir}")
  endforeach()

  add_custom_target(clang-test
    COMMAND sed -e "s#\@CLANG_SOURCE_DIR\@#${CMAKE_CURRENT_SOURCE_DIR}/..#"
                -e "s#\@CLANG_BINARY_DIR\@#${CMAKE_CURRENT_BINARY_DIR}/..#"
                -e "s#\@LLVM_TOOLS_DIR\@#${LLVM_TOOLS_PATH}/${CMAKE_CFG_INTDIR}#"
                ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in >
                ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    COMMAND ${PYTHON_EXECUTABLE} 
                ${LLVM_SOURCE_DIR}/utils/lit/lit.py
                -s ${CLANG_TEST_EXTRA_ARGS}
                ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS clang clang-cc index-test
                COMMENT "Running Clang regression tests")
endif()  
