set(LLVM_NO_RTTI 1)

add_clang_library(clangIndex
  ASTLocation.cpp
  DeclReferenceMap.cpp
  Entity.cpp
  IndexProvider.cpp
  Program.cpp
  ResolveLocation.cpp
  )
