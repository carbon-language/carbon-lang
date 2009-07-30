set(LLVM_NO_RTTI 1)

add_clang_library(clangIndex
  Analyzer.cpp
  ASTLocation.cpp
  DeclReferenceMap.cpp
  Entity.cpp
  GlobalSelector.cpp
  Handlers.cpp
  Indexer.cpp
  IndexProvider.cpp
  Program.cpp
  ResolveLocation.cpp
  SelectorMap.cpp
  )
