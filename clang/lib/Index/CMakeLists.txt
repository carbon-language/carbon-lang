set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS clangBasic clangAST)

add_clang_library(clangIndex
  ASTLocation.cpp
  Analyzer.cpp
  CallGraph.cpp
  DeclReferenceMap.cpp
  Entity.cpp
  GlobalSelector.cpp
  Handlers.cpp
  IndexProvider.cpp
  Indexer.cpp
  Program.cpp
  SelectorMap.cpp
  )

add_dependencies(clangIndex ClangAttrClasses ClangAttrList
                 ClangDeclNodes ClangStmtNodes)
