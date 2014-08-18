set(LLVM_LINK_COMPONENTS support)

add_clang_executable(clang-rename
  ClangRename.cpp
  USRFinder.cpp
  USRFindingAction.cpp
  USRLocFinder.cpp
  RenamingAction.cpp
  )

target_link_libraries(clang-rename
  clangAnalysis
  clangAST
  clangBasic
  clangDriver
  clangEdit
  clangFrontend
  clangFrontendTool
  clangIndex
  clangLex
  clangParse
  clangRewrite
  clangRewriteFrontend
  clangSerialization
  clangSema
  clangTooling
  )

install(TARGETS clang-rename RUNTIME DESTINATION bin)
