set(LLVM_LINK_COMPONENTS support)

add_clang_library(migrateCore
  FileOverrides.cpp
  SyntaxCheck.cpp
  Transforms.cpp
  Transform.cpp
  IncludeExcludeInfo.cpp
  PerfSupport.cpp
  Reformatting.cpp
  )
target_link_libraries(migrateCore
  clangFormat
  clangTooling
  clangBasic
  clangASTMatchers
  clangRewriteFrontend
  )
