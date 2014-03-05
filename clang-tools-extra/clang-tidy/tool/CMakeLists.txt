set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_executable(clang-tidy
  ClangTidyMain.cpp
  )
target_link_libraries(clang-tidy
  clangAST
  clangASTMatchers
  clangTidy
  clangTidyGoogleModule
  clangTidyLLVMModule
  clangTidyMiscModule
  clangTooling
  )

install(TARGETS clang-tidy
  RUNTIME DESTINATION bin)

