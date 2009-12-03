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
  ResolveLocation.cpp
  SelectorMap.cpp
  )
