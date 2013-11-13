add_clang_library(clangIndex
  CommentToXML.cpp
  SimpleFormatContext.h
  USRGeneration.cpp
  )

target_link_libraries(clangIndex
  clangBasic
  clangAST
  clangFormat
  )
