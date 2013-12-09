set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_executable(clang-tidy
  ClangTidyMain.cpp
  )
target_link_libraries(clang-tidy
  clangTidy
  clangTidyLLVMModule
  clangTidyGoogleModule
  clangStaticAnalyzerCheckers
  )

install(TARGETS clang-tidy
  RUNTIME DESTINATION bin)

