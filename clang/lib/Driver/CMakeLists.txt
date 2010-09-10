set(LLVM_NO_RTTI 1)

add_clang_library(clangDriver
  Action.cpp
  Arg.cpp
  ArgList.cpp
  CC1Options.cpp
  CC1AsOptions.cpp
  Compilation.cpp
  Driver.cpp
  DriverOptions.cpp
  HostInfo.cpp
  Job.cpp
  Option.cpp
  OptTable.cpp
  Phases.cpp
  Tool.cpp
  ToolChain.cpp
  ToolChains.cpp
  Tools.cpp
  Types.cpp
  )

target_link_libraries(clangDriver
  clangBasic
  clangAST
  clangParse
  )

add_dependencies(clangDriver
  ClangAttrList
  ClangDiagnosticDriver
  ClangDriverOptions
  ClangCC1Options
  ClangCC1AsOptions
  )
