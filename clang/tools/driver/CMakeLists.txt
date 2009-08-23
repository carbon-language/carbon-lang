set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  clangDriver
  clangBasic
  )

set(LLVM_LINK_COMPONENTS system support bitreader bitwriter)

add_clang_executable(clang
  driver.cpp
  )

add_dependencies(clang clang-cc)

install(TARGETS clang
  RUNTIME DESTINATION bin)
