set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangEdit
  Commit.cpp
  EditedSource.cpp
  RewriteObjCFoundationAPI.cpp
  )

target_link_libraries(clangEdit
  clangBasic
  clangAST
  clangLex
  )
