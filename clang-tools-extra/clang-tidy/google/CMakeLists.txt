set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyGoogleModule
  GoogleTidyModule.cpp
  )
target_link_libraries(clangTidyGoogleModule
  clangAST
  clangASTMatchers
  clangBasic
  clangTidy
  )
