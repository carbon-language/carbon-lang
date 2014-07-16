set(LLVM_LINK_COMPONENTS support)

add_clang_executable(clang-format
  ClangFormat.cpp
  )

target_link_libraries(clang-format
  clangBasic
  clangFormat
  clangLex
  clangRewrite
  clangTooling
  )

install(TARGETS clang-format RUNTIME DESTINATION bin)
install(PROGRAMS clang-format-bbedit.applescript DESTINATION share/clang)
install(PROGRAMS clang-format-diff.py DESTINATION share/clang)
install(PROGRAMS clang-format-sublime.py DESTINATION share/clang)
install(PROGRAMS clang-format.el DESTINATION share/clang)
install(PROGRAMS clang-format.py DESTINATION share/clang)
install(PROGRAMS git-clang-format DESTINATION bin)
