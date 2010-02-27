set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS
    clangFrontend
    clangDriver
    clangCodeGen
    clangSema
    clangChecker
    clangAnalysis
    clangRewrite
    clangAST
    clangParse
    clangLex
    clangBasic
    )

set(LLVM_LINK_COMPONENTS
    jit
    interpreter
    nativecodegen
    bitreader
    bitwriter
    ipo
    selectiondag
  )

add_clang_executable(clang-interpreter
  main.cpp
  )
add_dependencies(clang-interpreter clang-headers)
