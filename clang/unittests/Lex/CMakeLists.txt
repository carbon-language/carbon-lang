add_clang_unittest(LexTests
  LexerTest.cpp
  PPCallbacksTest.cpp
  PPConditionalDirectiveRecordTest.cpp
  )

target_link_libraries(LexTests
  clangLex clangParse clangSema
  )
