set(LLVM_TARGET_DEFINITIONS ../Basic/Attr.td)
tablegen(AttrSpellings.inc
         -gen-clang-attr-spelling-list
         -I ${CMAKE_CURRENT_SOURCE_DIR}/../../)
add_custom_target(ClangAttrSpellings
  DEPENDS AttrSpellings.inc)
