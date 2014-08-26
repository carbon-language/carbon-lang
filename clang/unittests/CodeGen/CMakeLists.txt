set(LLVM_LINK_COMPONENTS
  Support
  ProfileData
  )

add_clang_unittest(ClangCodeGenTests
  BufferSourceTest.cpp
  )

target_link_libraries(ClangCodeGenTests
  clangAST
  clangBasic
  clangFrontend
  clangParse
  clangSema
  clangTooling
  clangCodeGen
  )
