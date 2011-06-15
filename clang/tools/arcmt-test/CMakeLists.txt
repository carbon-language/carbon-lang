set(LLVM_USED_LIBS
  libclang
  clangARCMigrate
  clangRewrite
  )

set( LLVM_LINK_COMPONENTS
  support
  mc
  )

add_clang_executable(arcmt-test
  arcmt-test.cpp
  )
