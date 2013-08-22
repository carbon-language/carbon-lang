set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  support
  mc
  )

add_clang_library(clangReplace
  ApplyReplacements.cpp
  )
target_link_libraries(clangReplace
  clangTooling
  clangBasic
  clangRewriteFrontend
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(tool)
