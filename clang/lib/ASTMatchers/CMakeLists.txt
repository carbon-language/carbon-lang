add_subdirectory(Dynamic)

set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangASTMatchers
  ASTMatchFinder.cpp
  ASTMatchersInternal.cpp
  )

target_link_libraries(clangASTMatchers
  clangAST
  )
