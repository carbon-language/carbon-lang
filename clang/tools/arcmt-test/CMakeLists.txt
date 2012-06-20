set(LLVM_USED_LIBS
  clangARCMigrate
  clangEdit
  clangRewrite
  )

set( LLVM_LINK_COMPONENTS
  support
  mc
  )

add_clang_executable(arcmt-test
  arcmt-test.cpp
  )
