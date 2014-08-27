set(LLVM_LINK_COMPONENTS
  Core
  ProfileData
  Support
  )

add_clang_unittest(ClangCodeGenTests
  BufferSourceTest.cpp
  )

target_link_libraries(ClangCodeGenTests
  clangAST
  clangBasic
  clangCodeGen
  clangFrontend
  clangParse
  clangSema
  clangTooling
  )
