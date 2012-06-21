add_custom_target(ClangUnitTests)
set_target_properties(ClangUnitTests PROPERTIES FOLDER "Clang tests")

# add_clang_unittest(test_dirname file1.cpp file2.cpp)
#
# Will compile the list of files together and link against the clang
# Produces a binary named 'basename(test_dirname)'.
function(add_clang_unittest test_dirname)
  add_unittest(ClangUnitTests ${test_dirname} ${ARGN})
endfunction()

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
