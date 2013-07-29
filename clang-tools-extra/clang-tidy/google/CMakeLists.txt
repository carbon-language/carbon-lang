set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyGoogleModule
  GoogleTidyModule.cpp
  )
target_link_libraries(clangTidyGoogleModule
  clangTooling
  clangBasic
  clangASTMatchers
  )
