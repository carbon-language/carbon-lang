set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangDynamicASTMatchers
  Diagnostics.cpp
  VariantValue.cpp
  Parser.cpp
  Registry.cpp
  )

target_link_libraries(clangDynamicASTMatchers
  clangAST
  clangASTMatchers
  clangBasic
  )
