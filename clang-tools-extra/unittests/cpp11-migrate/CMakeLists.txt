set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CPP11_MIGRATE_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp11-migrate REALPATH)
include_directories(${CPP11_MIGRATE_SOURCE_DIR})

add_extra_unittest(Cpp11MigrateTests
  TransformTest.cpp
  IncludeExcludeTest.cpp)

target_link_libraries(Cpp11MigrateTests
  migrateCore
  clangTooling
  clangBasic
  clangASTMatchers
  )

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lit.local.cfg
  ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
