set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangIndex
  CommentToXML.cpp
  SimpleFormatContext.h
  USRGeneration.cpp
  )

target_link_libraries(clangIndex
  clangAST
  clangBasic
  clangFormat
  clangLex
  clangRewriteCore
  clangTooling
  )
