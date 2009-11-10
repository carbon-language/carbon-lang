set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  clangFrontend
  clangCodeGen
  clangAnalysis
  clangRewrite
  clangSema
  clangAST
  clangParse
  clangLex
  clangBasic
  )

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  bitreader
  bitwriter
  codegen
  ipo
  selectiondag
  )

add_clang_executable(clang-cc
  clang-cc.cpp
  Options.cpp
  )
add_dependencies(clang-cc clang-headers)

install(TARGETS clang-cc
  RUNTIME DESTINATION libexec)
