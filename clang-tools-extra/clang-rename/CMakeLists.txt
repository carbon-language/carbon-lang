set(LLVM_LINK_COMPONENTS support)

add_clang_executable(clang-rename
  ClangRename.cpp
  USRFinder.cpp
  USRFindingAction.cpp
  USRLocFinder.cpp
  RenamingAction.cpp
  )

target_link_libraries(clang-rename
  clangAST
  clangBasic
  clangFrontend
  clangIndex
  clangRewrite
  clangTooling
  )

install(TARGETS clang-rename RUNTIME DESTINATION bin)
