set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(LexTests
  LexerTest.cpp
  PPCallbacksTest.cpp
  PPConditionalDirectiveRecordTest.cpp
  )

target_link_libraries(LexTests
  clangAST
  clangBasic
  clangLex
  clangParse
  clangSema
  clangSerialization
  )
