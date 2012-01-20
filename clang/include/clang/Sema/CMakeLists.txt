clang_tablegen(AttrTemplateInstantiate.inc -gen-clang-attr-template-instantiate
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE ../Basic/Attr.td
  TARGET ClangAttrTemplateInstantiate)
