set(LLVM_NO_RTTI 1)

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

target_link_libraries(clangIndex
  clangSerialization
  clangFrontend
  clangParse
  clangSema
  clangChecker
  clangAnalysis
  clangIndex
  clangRewrite
  clangAST
  clangLex
  clangBasic
  )

add_dependencies(clangIndex
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangStmtNodes
  )
