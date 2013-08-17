add_clang_library(clangIndex
  USRGeneration.cpp
  )

target_link_libraries(clangIndex
  clangBasic
  clangAST
  )
