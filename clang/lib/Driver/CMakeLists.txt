set(LLVM_NO_RTTI 1)

add_clang_library(clangDriver
  HTMLDiagnostics.cpp
  InitHeaderSearch.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnosticPrinter.cpp
  )
