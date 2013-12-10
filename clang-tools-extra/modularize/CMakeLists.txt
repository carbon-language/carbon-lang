set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_executable(modularize
  Modularize.cpp
  ModuleAssistant.cpp
  PreprocessorTracker.cpp
  )

target_link_libraries(modularize
  clangBasic
  clangDriver
  clangFrontend
  clangLex
  clangTooling
  )
