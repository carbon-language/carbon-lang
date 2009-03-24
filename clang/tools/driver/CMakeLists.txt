set( LLVM_USED_LIBS
  clangDriver
  clangBasic
  )

set(LLVM_LINK_COMPONENTS system support bitreader bitwriter)

add_clang_tool(clang-driver
  driver.cpp
  )
