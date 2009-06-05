find_file(CLANG_TEST_RUNNER TestRunner.sh PATHS ${CMAKE_CURRENT_SOURCE_DIR}
  DOC "Clang's regression testing script")
mark_as_advanced(CLANG_TEST_RUNNER)

get_target_property(LLVM_TOOLS_PATH clang RUNTIME_OUTPUT_DIRECTORY)

add_custom_target(clang-test COMMENT "Running Clang regression tests")

macro(add_clang_test_suite language target extension)
add_custom_target(${target}
  ${CMAKE_CTEST_COMMAND}
  --build-and-test
  ${LLVM_SOURCE_DIR}/tools/clang/CMake/RunTests
  ${CMAKE_CURRENT_BINARY_DIR}/${target}
  --build-generator ${CMAKE_GENERATOR}
  --build-makeprogram ${CMAKE_MAKE_PROGRAM}
  --build-project ClangTest
  --build-target test
  --build-options 
  "-DCLANG_TEST_RUNNER=${CLANG_TEST_RUNNER}"
  "-DCLANG_TEST_GLOB_PATTERN=${CMAKE_CURRENT_SOURCE_DIR}/*.${extension}"
  "-DLLVM_TOOLS_PATH=${LLVM_TOOLS_PATH}"
  "-DLLVM_SCRIPTS_PATH=${LLVM_SOURCE_DIR}/test/Scripts"
  COMMENT "Running Clang ${language} regression tests")

  add_dependencies(clang-test ${target})
endmacro(add_clang_test_suite)

add_clang_test_suite(C clang-test-c c)
add_clang_test_suite(Objective-C clang-test-objc m)
add_clang_test_suite(C++ clang-test-cxx cpp)
add_clang_test_suite(Objective-C++ clang-test-objcxx mm)
add_clang_test_suite(Assembler clang-test-asm S)
