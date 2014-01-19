set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(FormatTests
  FormatTest.cpp
  FormatTestJS.cpp
  FormatTestProto.cpp
  )

target_link_libraries(FormatTests
  clangFormat
  clangTooling
  )
