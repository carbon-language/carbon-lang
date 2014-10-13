add_clang_executable(clang-rename ClangRename.cpp)

target_link_libraries(clang-rename
  clangAST
  clangBasic
  clangFrontend
  clangIndex
  clangRename
  clangRewrite
  clangTooling)

install(TARGETS clang-rename RUNTIME DESTINATION bin)
