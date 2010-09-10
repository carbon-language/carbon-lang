set(LLVM_NO_RTTI 1)

add_clang_executable(clang-interpreter
  main.cpp
  )

add_llvm_link_components(clang-interpreter jit interpreter nativecodegen)
target_link_libraries(clang-interpreter
  clangFrontend
  clangSerialization
  clangDriver
  clangCodeGen
  clangSema
  clangChecker
  clangIndex
  clangAnalysis
  clangRewrite
  clangAST
  clangParse
  clangLex
  clangBasic
  )

add_dependencies(clang-interpreter clang-headers)
