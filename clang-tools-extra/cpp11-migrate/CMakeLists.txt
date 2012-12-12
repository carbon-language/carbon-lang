set(LLVM_LINK_COMPONENTS support)
set(LLVM_USED_LIBS clangTooling clangBasic clangAST)

add_clang_executable(cpp11-migrate
  Cpp11Migrate.cpp
  )

target_link_libraries(cpp11-migrate
  clangTooling
  clangBasic
  clangASTMatchers
  )
