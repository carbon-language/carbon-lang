add_clang_executable(clang-rename ClangRename.cpp)

target_link_libraries(clang-rename
  clangBasic
  clangFrontend
  clangRename
  clangRewrite
  clangTooling
  )

install(TARGETS clang-rename RUNTIME DESTINATION bin)
