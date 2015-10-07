set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(ClangDriverTests
  ToolChainTest.cpp
  MultilibTest.cpp
  )

target_link_libraries(ClangDriverTests
  clangDriver
  clangBasic
  )
