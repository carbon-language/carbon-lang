add_clang_library(clangIDE
  USRGeneration.cpp
  )

target_link_libraries(clangIDE
  clangBasic
  clangAST
  )
