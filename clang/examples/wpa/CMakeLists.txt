set(LLVM_NO_RTTI 1)

add_clang_executable(clang-wpa
  clang-wpa.cpp
  )

target_link_libraries(clang-wpa
  clangIndex
  clangFrontend
  clangDriver
  clangSema
  clangAnalysis
  clangSerialization
  clangChecker
  clangRewrite
  clangAST
  clangParse
  clangLex
  clangBasic
  )

add_dependencies(clang-wpa
  clang-headers
  )
