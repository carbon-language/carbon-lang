set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_library(clangDriver
  Action.cpp
  Compilation.cpp
  Driver.cpp
  DriverOptions.cpp
  Job.cpp
  Multilib.cpp
  Phases.cpp
  SanitizerArgs.cpp
  Tool.cpp
  ToolChain.cpp
  ToolChains.cpp
  WindowsToolChain.cpp
  Tools.cpp
  Types.cpp

  DEPENDS
  ClangDriverOptions

  LINK_LIBS
  clangBasic
  )
