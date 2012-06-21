add_clang_unittest(LexTests
  LexerTest.cpp
  PreprocessingRecordTest.cpp
  )

target_link_libraries(LexTests
  clangLex
  )
