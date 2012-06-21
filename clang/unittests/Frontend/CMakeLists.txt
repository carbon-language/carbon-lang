add_clang_unittest(FrontendTests
  FrontendActionTest.cpp
  )
target_link_libraries(FrontendTests
  clangFrontend
  )
