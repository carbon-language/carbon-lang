set(LLVM_NO_RTTI 1)

add_clang_library(clangDriver
  Action.cpp
  Arg.cpp
  ArgList.cpp
  CC1Options.cpp
  Compilation.cpp
  Driver.cpp
  DriverOptions.cpp
  HostInfo.cpp
  Job.cpp
  OptTable.cpp
  Option.cpp
  Phases.cpp
  Tool.cpp
  ToolChain.cpp
  ToolChains.cpp
  Tools.cpp
  Types.cpp
  )

add_dependencies(clangDriver ClangDiagnosticDriver ClangDriverOptions ClangCC1Options)
