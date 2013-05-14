set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangDynamicASTMatchers
  Diagnostics.cpp
  VariantValue.cpp
  Parser.cpp
  Registry.cpp
  )

add_dependencies(clangDynamicASTMatchers
  clangASTMatchers
  )

target_link_libraries(clangDynamicASTMatchers
  clangASTMatchers
  )
