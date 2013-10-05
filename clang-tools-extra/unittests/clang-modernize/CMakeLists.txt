set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CLANG_MODERNIZE_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-modernize REALPATH)
get_filename_component(ClangReplaceLocation
  "${CMAKE_CURRENT_SOURCE_DIR}/../../clang-apply-replacements/include" REALPATH)
get_filename_component(CommonIncLocation
  "${CMAKE_CURRENT_SOURCE_DIR}/../include" REALPATH)
include_directories(
  ${CLANG_MODERNIZE_SOURCE_DIR}
  ${ClangReplaceLocation}
  ${CommonIncLocation}
  )

add_extra_unittest(ClangModernizeTests
  IncludeExcludeTest.cpp
  PerfSupportTest.cpp
  TransformTest.cpp
  UniqueHeaderNameTest.cpp
  IncludeDirectivesTest.cpp
  )

target_link_libraries(ClangModernizeTests
  modernizeCore
  clangFormat
  clangTooling
  clangBasic
  clangASTMatchers
  clangRewriteFrontend
  )
