set(LLVM_LINK_COMPONENTS support)
set(LLVM_USED_LIBS clangFormat clangTooling clangBasic clangAST)

add_clang_executable(clang-format
  ClangFormat.cpp
  )

target_link_libraries(clang-format
  clangFormat
  clangTooling
  clangBasic
  clangRewriteFrontend
  )
