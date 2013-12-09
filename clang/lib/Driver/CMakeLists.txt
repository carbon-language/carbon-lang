set(LLVM_LINK_COMPONENTS
  Option
  Support
  TransformUtils
  )

add_clang_library(clangDriver
  Action.cpp
  CC1AsOptions.cpp
  Compilation.cpp
  Driver.cpp
  DriverOptions.cpp
  Job.cpp
  Phases.cpp
  SanitizerArgs.cpp
  Tool.cpp
  ToolChain.cpp
  ToolChains.cpp
  WindowsToolChain.cpp
  Tools.cpp
  Types.cpp
  )

add_dependencies(clangDriver
  ClangAttrList
  ClangCC1AsOptions
  ClangDiagnosticCommon
  ClangDiagnosticDriver
  ClangDriverOptions
  )

target_link_libraries(clangDriver
  clangBasic
  )
