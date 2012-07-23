add_clang_unittest(ASTTests
  CommentLexer.cpp
  CommentParser.cpp
  )

target_link_libraries(ASTTests
  clangAST
  )
