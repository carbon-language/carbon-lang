set(LLVM_NO_RTTI 1)

add_clang_library(clangDriver
  Action.cpp
  Arg.cpp
  ArgList.cpp
  Compilation.cpp
  Driver.cpp
  HostInfo.cpp
  OptTable.cpp
  Option.cpp
  Phases.cpp
  Types.cpp
  )
