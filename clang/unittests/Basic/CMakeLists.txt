set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(BasicTests
  CharInfoTest.cpp
  FileManagerTest.cpp
  SourceManagerTest.cpp
  )

target_link_libraries(BasicTests
  clangBasic
  clangLex
  )
