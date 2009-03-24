set(LLVM_NO_RTTI 1)

add_clang_library(clangFrontend
  HTMLDiagnostics.cpp
  InitHeaderSearch.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnosticPrinter.cpp
  PlistDiagnostics.cpp
  ManagerRegistry.cpp
  )
