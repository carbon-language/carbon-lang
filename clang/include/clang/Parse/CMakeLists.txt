clang_tablegen(AttrParserStringSwitches.inc -gen-clang-attr-parser-string-switches
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrParserStringSwitches)
