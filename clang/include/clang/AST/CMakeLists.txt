set(LLVM_TARGET_DEFINITIONS StmtNodes.td)
tablegen(StmtNodes.inc
         -gen-clang-stmt-nodes)
add_custom_target(ClangStmtNodes
  DEPENDS StmtNodes.inc)
