set(LLVM_LINK_COMPONENTS support)

add_clang_library(modernizeCore
  FileOverrides.cpp
  SyntaxCheck.cpp
  Transforms.cpp
  Transform.cpp
  IncludeExcludeInfo.cpp
  PerfSupport.cpp
  Reformatting.cpp
  IncludeDirectives.cpp
  )
target_link_libraries(modernizeCore
  clangFormat
  clangTooling
  clangBasic
  clangASTMatchers
  clangRewriteFrontend
  )
