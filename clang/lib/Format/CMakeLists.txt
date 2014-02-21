set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangFormat
  BreakableToken.cpp
  ContinuationIndenter.cpp
  Format.cpp
  FormatToken.cpp
  TokenAnnotator.cpp
  UnwrappedLineParser.cpp
  WhitespaceManager.cpp
  )

target_link_libraries(clangFormat
  clangBasic
  clangLex
  clangTooling
  )
