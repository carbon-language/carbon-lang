set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangApplyReplacements
  lib/Tooling/ApplyReplacements.cpp
  )
target_link_libraries(clangApplyReplacements
  clangBasic
  clangRewriteCore
  clangTooling
  )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  include
  )
add_subdirectory(tool)
