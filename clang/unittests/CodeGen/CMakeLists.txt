set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(CodeGenTests
  BufferSourceTest.cpp
  )

target_link_libraries(CodeGenTests
  clangAST
  clangBasic
  clangFrontend
  clangParse
  clangSema
  clangTooling
  clangCodeGen
  )
