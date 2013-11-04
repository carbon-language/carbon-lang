clang_tablegen(AttrIdentifierArg.inc -gen-clang-attr-identifier-arg-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrIdentifierArg)

clang_tablegen(AttrTypeArg.inc -gen-clang-attr-type-arg-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrTypeArg)

clang_tablegen(AttrLateParsed.inc -gen-clang-attr-late-parsed-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrLateParsed)
