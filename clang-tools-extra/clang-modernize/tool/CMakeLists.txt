set(LLVM_LINK_COMPONENTS support)

set (ClangModernizeSources
  ClangModernize.cpp
  )

# FIXME: Lib-ify the transforms to simplify the build rules.

# For each transform subdirectory.
file(GLOB_RECURSE LoopConvertSources "../LoopConvert/*.cpp")
list(APPEND ClangModernizeSources ${LoopConvertSources})

file(GLOB_RECURSE UseNullptrSources "../UseNullptr/*.cpp")
list(APPEND ClangModernizeSources ${UseNullptrSources})

file(GLOB_RECURSE UseAutoSources "../UseAuto/*.cpp")
list(APPEND ClangModernizeSources ${UseAutoSources})

file(GLOB_RECURSE AddOverrideSources "../AddOverride/*.cpp")
list(APPEND ClangModernizeSources ${AddOverrideSources})

file(GLOB_RECURSE PassByValueSources "../PassByValue/*.cpp")
list(APPEND ClangModernizeSources ${PassByValueSources})

file(GLOB_RECURSE ReplaceAutoPtrSources "../ReplaceAutoPtr/*.cpp")
list(APPEND ClangModernizeSources ${ReplaceAutoPtrSources})

add_clang_executable(clang-modernize
  ${ClangModernizeSources}
  )

add_dependencies(clang-modernize
  clang-headers clang-apply-replacements
  )

target_link_libraries(clang-modernize
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangTooling
  modernizeCore
  )

install(TARGETS clang-modernize
  RUNTIME DESTINATION bin)
