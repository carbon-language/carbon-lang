add_clang_unittest(ASTMatchersTests
  ASTMatchersTest.cpp)

target_link_libraries(ASTMatchersTests
  gtest gtest_main clangASTMatchers clangTooling)
