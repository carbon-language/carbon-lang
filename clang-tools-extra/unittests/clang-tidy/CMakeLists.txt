set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CLANG_LINT_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-tidy REALPATH)
include_directories(${CLANG_LINT_SOURCE_DIR})

add_extra_unittest(ClangTidyTests
  ClangTidyDiagnosticConsumerTest.cpp
  ClangTidyOptionsTest.cpp
  GoogleModuleTest.cpp
  LLVMModuleTest.cpp
  MiscModuleTest.cpp
  ReadabilityModuleTest.cpp)

target_link_libraries(ClangTidyTests
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTidy
  clangTidyGoogleModule
  clangTidyLLVMModule
  clangTidyMiscModule
  clangTidyReadabilityModule
  clangTidyUtils
  clangTooling
  clangToolingCore
  )
