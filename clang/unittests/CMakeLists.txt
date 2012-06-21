# add_clang_unittest(test_dirname file1.cpp file2.cpp)
#
# Will compile the list of files together and link against the clang
# Produces a binary named 'basename(test_dirname)'.
function(add_clang_unittest test_dirname)
  string(REGEX MATCH "([^/]+)$" test_name ${test_dirname})
  if (CMAKE_BUILD_TYPE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${CLANG_BINARY_DIR}/unittests/${test_dirname}/${CMAKE_BUILD_TYPE})
  else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${CLANG_BINARY_DIR}/unittests/${test_dirname})
  endif()
  if( NOT LLVM_BUILD_TESTS )
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_clang_executable(${test_name} ${ARGN})
  target_link_libraries(${test_name}
    gtest
    gtest_main
    LLVMSupport # gtest needs it for raw_ostream.
    )

  add_dependencies(ClangUnitTests ${test_name})
  set_target_properties(${test_name} PROPERTIES FOLDER "Clang tests")
endfunction()

add_custom_target(ClangUnitTests)
set_target_properties(ClangUnitTests PROPERTIES FOLDER "Clang tests")

include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)

add_definitions(-DGTEST_HAS_RTTI=0)
if( LLVM_COMPILER_IS_GCC_COMPATIBLE )
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS "-frtti" "-fno-rtti")
elseif( MSVC )
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS "/GR" "/GR-")
endif()

if (NOT LLVM_ENABLE_THREADS)
  add_definitions(-DGTEST_HAS_PTHREAD=0)
endif()

if(SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  add_definitions("-Wno-variadic-macros")
endif()

add_clang_unittest(BasicTests
  Basic/FileManagerTest.cpp
  Basic/SourceManagerTest.cpp
  )
target_link_libraries(BasicTests
  clangLex
  )

add_clang_unittest(LexTests
  Lex/LexerTest.cpp
  )
target_link_libraries(LexTests
  clangLex
  )

add_clang_unittest(FrontendTests
  Frontend/FrontendActionTest.cpp
  )
target_link_libraries(FrontendTests
  clangFrontend
  )

add_clang_unittest(ToolingTests
  Tooling/CompilationDatabaseTest.cpp
  Tooling/ToolingTest.cpp
  Tooling/RecursiveASTVisitorTest.cpp
  Tooling/RefactoringTest.cpp
  Tooling/RewriterTest.cpp
  )
target_link_libraries(ToolingTests
  clangAST
  clangTooling
  clangRewrite
  )
