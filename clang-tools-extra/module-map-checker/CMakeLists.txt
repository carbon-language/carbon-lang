set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  mc
  )

add_clang_executable(module-map-checker
  ModuleMapChecker.cpp
  )

target_link_libraries(module-map-checker
  clangTooling
  clangBasic
  clangRewriteFrontend
  )
