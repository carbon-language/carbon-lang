set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CPP11_MIGRATE_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp11-migrate REALPATH)
get_filename_component(ClangReplaceLocation
  "${CMAKE_CURRENT_SOURCE_DIR}/../../clang-replace/include" REALPATH)
include_directories(
  ${CPP11_MIGRATE_SOURCE_DIR}
  ${ClangReplaceLocation}
  )

add_extra_unittest(Cpp11MigrateTests
  FileOverridesTest.cpp
  ReformattingTest.cpp
  IncludeExcludeTest.cpp
  PerfSupportTest.cpp
  TransformTest.cpp
  UniqueHeaderNameTest.cpp
  IncludeDirectivesTest.cpp
  )

target_link_libraries(Cpp11MigrateTests
  migrateCore
  clangFormat
  clangTooling
  clangBasic
  clangASTMatchers
  clangRewriteFrontend
  )
