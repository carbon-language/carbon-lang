set(LLVM_TARGET_DEFINITIONS StmtNodes.td)
tablegen(StmtNodes.inc
         -gen-clang-stmt-nodes)
add_custom_target(ClangStmtNodes
  DEPENDS StmtNodes.inc)

set(LLVM_TARGET_DEFINITIONS DeclNodes.td)
tablegen(DeclNodes.inc
         -gen-clang-decl-nodes)
add_custom_target(ClangDeclNodes
  DEPENDS DeclNodes.inc)
