set(LLVM_LINK_COMPONENTS support)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (Cpp11MigrateSources
  Cpp11Migrate.cpp
  Transforms.cpp
  )

# For each transform subdirectory.
file(GLOB_RECURSE LoopConvertSources "LoopConvert/*.cpp")
list(APPEND Cpp11MigrateSources ${LoopConvertSources})

add_clang_executable(cpp11-migrate
  ${Cpp11MigrateSources}
  )

target_link_libraries(cpp11-migrate
  clangTooling
  clangBasic
  clangASTMatchers
  )
