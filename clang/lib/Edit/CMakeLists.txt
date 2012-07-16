add_clang_library(clangEdit
  Commit.cpp
  EditedSource.cpp
  RewriteObjCFoundationAPI.cpp
  )

add_dependencies(clangEdit
  ClangDiagnosticCommon
)

target_link_libraries(clangEdit
  clangBasic
  clangAST
  clangLex
  )
