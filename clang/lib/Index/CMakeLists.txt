set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangIndex
  CommentToXML.cpp
  USRGeneration.cpp

  ADDITIONAL_HEADERS
  SimpleFormatContext.h
  )

target_link_libraries(clangIndex
  clangAST
  clangBasic
  clangFormat
  clangLex
  clangRewriteCore
  clangTooling
  )
