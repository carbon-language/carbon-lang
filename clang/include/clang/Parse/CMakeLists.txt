clang_tablegen(AttrExprArgs.inc -gen-clang-attr-expr-args-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrExprArgs)

clang_tablegen(AttrLateParsed.inc -gen-clang-attr-late-parsed-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrLateParsed)
