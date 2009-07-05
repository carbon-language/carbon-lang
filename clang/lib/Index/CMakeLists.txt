set(LLVM_NO_RTTI 1)

add_clang_library(clangIndex
  Entity.cpp
  IndexProvider.cpp
  Program.cpp
  )
