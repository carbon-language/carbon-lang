set(LLVM_REQUIRES_EH 1)
set(LLVM_REQUIRES_RTTI 1)
set(LLVM_LINK_COMPONENTS Support)

add_tablegen(clang-tblgen CLANG
  ClangASTNodesEmitter.cpp
  ClangAttrEmitter.cpp
  ClangDiagnosticsEmitter.cpp
  ClangSACheckersEmitter.cpp
  NeonEmitter.cpp
  OptParserEmitter.cpp
  TableGen.cpp
  )
