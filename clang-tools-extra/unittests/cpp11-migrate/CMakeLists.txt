set(LLVM_LINK_COMPONENTS
  support
  )

add_extra_unittest(Cpp11MigrateTests
  dummy.cpp)

target_link_libraries(Cpp11MigrateTests
  clangTooling
  clangBasic
  clangASTMatchers
  )

