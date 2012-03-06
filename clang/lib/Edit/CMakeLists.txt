set(LLVM_USED_LIBS clangBasic clangAST clangLex)

add_clang_library(clangEdit
  Commit.cpp
  EditedSource.cpp
  RewriteObjCFoundationAPI.cpp
  )
