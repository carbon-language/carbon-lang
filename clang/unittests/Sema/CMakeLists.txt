add_clang_unittest(SemaTests
  ExternalSemaSourceTest.cpp
  )

target_link_libraries(SemaTests
  clangAST clangASTMatchers clangTooling
  )
