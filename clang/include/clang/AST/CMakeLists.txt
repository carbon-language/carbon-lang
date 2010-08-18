set(LLVM_TARGET_DEFINITIONS ../Basic/Attr.td)
tablegen(Attrs.inc
         -gen-clang-attr-classes
         -I ${CMAKE_CURRENT_SOURCE_DIR}/../../)
add_custom_target(ClangAttrClasses
  DEPENDS Attrs.inc)

tablegen(AttrImpl.inc
         -gen-clang-attr-impl
         -I ${CMAKE_CURRENT_SOURCE_DIR}/../../)
add_custom_target(ClangAttrImpl
  DEPENDS AttrImpl.inc)

set(LLVM_TARGET_DEFINITIONS ../Basic/StmtNodes.td)
tablegen(StmtNodes.inc
         -gen-clang-stmt-nodes)
add_custom_target(ClangStmtNodes
  DEPENDS StmtNodes.inc)

set(LLVM_TARGET_DEFINITIONS ../Basic/DeclNodes.td)
tablegen(DeclNodes.inc
         -gen-clang-decl-nodes)
add_custom_target(ClangDeclNodes
  DEPENDS DeclNodes.inc)
