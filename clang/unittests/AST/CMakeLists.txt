add_clang_unittest(ASTTests
  CommentLexer.cpp
  )

target_link_libraries(ASTTests
  clangAST
  )
