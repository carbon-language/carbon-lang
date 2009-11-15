set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  clangFrontend
  clangAnalysis
  clangSema
  clangAST
  clangLex
  clangBasic
  clangIndex
  )

set( LLVM_LINK_COMPONENTS
    mc
  )

add_clang_executable(clang-wpa
  clang-wpa.cpp
  )
add_dependencies(clang-wpa clang-headers)
