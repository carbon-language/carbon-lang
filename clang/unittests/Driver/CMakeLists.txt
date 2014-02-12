set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(DriverTests
  MultilibTest.cpp
  )

target_link_libraries(DriverTests
  clangDriver
  )
