set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  support
  mc
  )

add_clang_unittest(FormatTests
  FormatTest.cpp
  )

target_link_libraries(FormatTests
  clangAST
  clangFormat
  clangTooling
  clangRewriteCore
  )
