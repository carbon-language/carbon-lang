set(LLVM_USED_LIBS clangBasic clangAST clangLex)

add_clang_library(clangComments
  RawCommentList.cpp
  )

